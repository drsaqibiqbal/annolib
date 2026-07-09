"""Real DICOM file I/O: read a .dcm into a Sample, write a Sample back out.

This is the bridge between the Transform/Compose architecture (which operates
on a Sample) and actual DICOM files on disk. Without it, the library is only
an in-memory abstraction; with it, `load_sample -> pipeline -> save_sample`
is a complete de-identification of a real study.

Design notes:
  * `dicom_meta` is populated as a plain {keyword: str-value} dict so the
    existing Transforms keep working unchanged, but the originating pydicom
    Dataset is retained on `Sample.source_dataset` so deletions/edits can be
    reflected back into the real object on save.
  * Only elements the Transforms actually reason about need to appear in the
    dict; private elements are surfaced under their `(gggg,eeee)` tag string
    so DicomTagScrub can see and strip them.
  * Pixel data is decoded to a NumPy array for the burned-in-text and mask
    transforms, and written back only if a Transform marked the image dirty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .core import Sample

# Keywords whose values are plain identifiers we surface to the Transforms.
# This is not the removal list (that lives in profiles.py) -- it is simply
# which header values we expose in dicom_meta for inspection/redaction.
_EXPOSED_KEYWORDS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "PatientAge",
    "OtherPatientIDs",
    "OtherPatientNames",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "InstitutionName",
    "InstitutionAddress",
    "StationName",
    "StudyID",
    "AccessionNumber",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
    "Manufacturer",
    "BurnedInAnnotation",
]


def load_sample(path: str | Path) -> Sample:
    """Read a DICOM file into a Sample.

    The full pydicom Dataset is kept on `sample.source_dataset`; a
    keyword->value view of the elements the Transforms care about (plus any
    private elements, keyed by their tag string) is placed in
    `sample.dicom_meta`.
    """
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)

    dicom_meta: Dict[str, Any] = {}
    for keyword in _EXPOSED_KEYWORDS:
        if keyword in ds:
            dicom_meta[keyword] = str(ds.data_element(keyword).value)

    # Surface private elements by tag string so DicomTagScrub can strip them.
    for elem in ds:
        if elem.tag.is_private:
            dicom_meta[f"({elem.tag.group:04X},{elem.tag.element:04X})"] = "<private>"

    image = _extract_pixel_array(ds)

    return Sample(image=image, dicom_meta=dicom_meta, source_dataset=ds)


def _extract_pixel_array(ds: Any) -> np.ndarray:
    """Decode pixel data to a CZYX float array, or a zero placeholder.

    Kept defensive: a header-only object (no PixelData) yields a tiny
    placeholder so header-scrubbing pipelines still run.
    """
    if "PixelData" not in ds:
        return np.zeros((1, 1, 1, 1), dtype=np.float32)

    arr = ds.pixel_array.astype(np.float32)
    # Normalise to CZYX so the mask/redaction transforms have a stable layout.
    if arr.ndim == 2:  # single 2D frame (H, W)
        arr = arr[np.newaxis, np.newaxis, ...]
    elif arr.ndim == 3:  # multi-frame or RGB (F, H, W) -> treat F as Z
        arr = arr[np.newaxis, ...]
    return arr


def save_sample(sample: Sample, path: str | Path) -> Path:
    """Write a (de-identified) Sample back to a .dcm file.

    Reconciles the Transforms' changes into `sample.source_dataset`:
      * any exposed keyword deleted from dicom_meta is removed from the ds;
      * any private tag deleted from dicom_meta is removed from the ds;
      * pseudonymised keyword values are written back;
      * pixel data is re-encoded only if a Transform set image_dirty.
    """
    path = Path(path)
    if sample.source_dataset is None:
        raise ValueError(
            "save_sample requires a Sample loaded via load_sample "
            "(sample.source_dataset is None)."
        )

    ds = sample.source_dataset
    meta = sample.dicom_meta or {}

    # 1. Delete exposed keywords that the Transforms removed from the dict.
    for keyword in _EXPOSED_KEYWORDS:
        if keyword in ds and keyword not in meta:
            delattr(ds, keyword)
        elif keyword in ds and keyword in meta:
            # Write back a possibly-pseudonymised value.
            current = str(ds.data_element(keyword).value)
            if current != meta[keyword]:
                try:
                    ds.data_element(keyword).value = meta[keyword]
                except Exception:
                    pass

    # 2. Delete private tags the Transforms removed.
    for elem in list(ds):
        if elem.tag.is_private:
            key = f"({elem.tag.group:04X},{elem.tag.element:04X})"
            if key not in meta:
                del ds[elem.tag]

    # 3. Re-encode pixel data only if it was actually modified.
    if sample.image_dirty and "PixelData" in ds:
        _write_pixel_array(ds, sample.image)

    # 4. Keep file-meta UID in sync if SOPInstanceUID was regenerated,
    #    otherwise the file is internally inconsistent (dataset vs. file meta).
    if "SOPInstanceUID" in ds and getattr(ds, "file_meta", None) is not None:
        try:
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        except Exception:
            pass

    # 5. Mark the file as de-identified per DICOM PS3.15 convention.
    try:
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "prostate-deid pipeline"
    except Exception:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path))
    return path


def _write_pixel_array(ds: Any, image: np.ndarray) -> None:
    """Write a CZYX float image back into ds.PixelData in its native dtype."""
    frame = np.squeeze(image)
    original = ds.pixel_array
    frame = frame.reshape(original.shape).astype(original.dtype)
    ds.PixelData = frame.tobytes()
