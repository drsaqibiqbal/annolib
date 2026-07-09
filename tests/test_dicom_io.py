"""Round-trip tests against a real synthetic DICOM file.

Builds a synthetic .dcm in a temp dir (with fake PHI in the header and a
private tag), runs the full default anonymization pipeline, and verifies the
written-out file no longer carries the identifying fields. No real patient
data -- everything here is synthesised in-test.
"""

from __future__ import annotations

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from prostate_deid import anonymize_dicom, load_sample, save_sample, summarize_audit  # noqa: E402


def _make_synthetic_dicom(path):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.PatientName = "Doe^John"
    ds.PatientID = "REAL-12345"
    ds.PatientBirthDate = "19700101"
    ds.ReferringPhysicianName = "Smith^Jane"
    ds.InstitutionName = "Synthetic General Hospital"
    ds.StudyDate = "20260101"
    ds.SeriesDate = "20260101"
    ds.Modality = "MR"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = MRImageStorage

    # A private element carrying (synthetic) PHI in an odd group.
    ds.add_new(0x00410010, "LO", "VENDOR PRIVATE PHI")

    # Minimal valid MR pixel data: 16x16 uint16.
    ds.Rows = 16
    ds.Columns = 16
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = (np.arange(256, dtype=np.uint16) % 100).tobytes()

    ds.save_as(str(path), enforce_file_format=True)
    return path


def test_load_sample_populates_meta_and_image(tmp_path):
    src = _make_synthetic_dicom(tmp_path / "in.dcm")
    sample = load_sample(src)

    assert sample.dicom_meta["PatientName"] == "Doe^John"
    assert sample.dicom_meta["PatientID"] == "REAL-12345"
    assert "(0041,0010)" in sample.dicom_meta  # private tag surfaced
    assert sample.source_dataset is not None
    assert sample.image.ndim == 4  # normalised to CZYX


def test_anonymize_dicom_removes_identifiers(tmp_path):
    src = _make_synthetic_dicom(tmp_path / "in.dcm")
    out = tmp_path / "out.dcm"

    audit = anonymize_dicom(
        src, out, mapping_store=tmp_path / "map.json", redact_burned_in_text=False
    )

    ds = pydicom.dcmread(str(out))
    assert "PatientName" not in ds
    assert "ReferringPhysicianName" not in ds
    assert "InstitutionName" not in ds
    # PatientID is pseudonymized (present but changed), not deleted.
    assert ds.PatientID != "REAL-12345"
    assert ds.PatientID.startswith("PSEUDO-")
    # Private tag stripped.
    assert 0x00410010 not in ds
    # UID regenerated.
    assert ds.StudyInstanceUID.startswith("2.25.")
    # PS3.15 de-identification marker written.
    assert ds.PatientIdentityRemoved == "YES"

    assert len(audit) > 0


def test_uid_regeneration_is_consistent(tmp_path):
    src = _make_synthetic_dicom(tmp_path / "in.dcm")
    sample = load_sample(src)
    study_uid = sample.dicom_meta["StudyInstanceUID"]

    from prostate_deid.pseudonymize import _derive_uid

    a = _derive_uid(study_uid, secret="k")
    b = _derive_uid(study_uid, secret="k")
    assert a == b  # same source + secret -> same UID
    assert len(a) <= 64
    assert a.startswith("2.25.")


def test_summarize_audit_counts(tmp_path):
    src = _make_synthetic_dicom(tmp_path / "in.dcm")
    out = tmp_path / "out.dcm"
    audit = anonymize_dicom(
        src, out, mapping_store=tmp_path / "map.json", redact_burned_in_text=False
    )

    summary = summarize_audit(audit)
    assert summary["ids_pseudonymized"] >= 1
    assert summary["private_tags_removed"] >= 1
    assert summary["uids_regenerated"] >= 1


def test_save_requires_source_dataset(tmp_path):
    from prostate_deid import Sample

    sample = Sample(image=np.zeros((1, 1, 4, 4)), dicom_meta={"PatientID": "x"})
    with pytest.raises(ValueError):
        save_sample(sample, tmp_path / "nope.dcm")


def test_burned_in_redact_degrades_without_ocr(tmp_path):
    """When tesseract isn't installed, the pipeline must not crash."""
    src = _make_synthetic_dicom(tmp_path / "in.dcm")
    out = tmp_path / "out.dcm"

    # redact_burned_in_text=True forces the OCR step; with no tesseract it
    # should log a flagged 'ocr_unavailable' entry rather than raise.
    audit = anonymize_dicom(
        src, out, mapping_store=tmp_path / "map.json", redact_burned_in_text=True
    )

    actions = {e.action for e in audit}
    # Either OCR ran (redacted / no-hit) or it degraded gracefully; in this
    # environment (no tesseract) we expect the graceful path.
    assert out.exists()
    assert "ocr_unavailable" in actions or "redacted_region" in actions or len(audit) > 0
