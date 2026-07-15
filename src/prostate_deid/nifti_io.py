"""Self-contained NIfTI-1 writer (numpy only, no external dependency).

Converts a Sample's volumetric image to a single-file .nii, deriving the
spatial affine from DICOM geometry when the Sample was loaded from a real
.dcm. NIfTI is a natural de-identification-adjacent export target for
research pipelines: the NIfTI-1 header carries essentially no patient
metadata (unlike a DICOM header), so writing to NIfTI discards the header
PHI surface entirely -- the pixels and geometry are all that survive.

The NIfTI-1 header is a fixed 348-byte little-endian struct; this module
writes it directly rather than depending on nibabel, so DICOM->NIfTI works
with only numpy installed. The output is validated against nibabel in the
test suite.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .core import Sample

_NIFTI1_HEADER_SIZE = 348
_VOX_OFFSET = 352.0  # header (348) padded to 352 before the data block
_DT_FLOAT32 = 16


def _single_volume(image: np.ndarray) -> np.ndarray:
    """Reduce a CZYX(T) Sample image to a 3-D (Z, Y, X) volume."""
    arr = np.asarray(image, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # (1, Y, X)
    if arr.ndim > 3:
        # Collapse any leading channel/time axes by taking the first index.
        while arr.ndim > 3:
            arr = arr[0]
    return arr  # (Z, Y, X)


def _affine_from_dataset(ds: Any) -> np.ndarray:
    """Build a basic RAS-oriented affine from DICOM spacing/position.

    DICOM PixelSpacing is [row spacing (y), col spacing (x)]. DICOM patient
    space is LPS; NIfTI prefers RAS, so the x and y axes are negated. This is
    a diagonal (axis-aligned) approximation -- adequate for de-identification/
    preprocessing where exact oblique orientation is not the point, and
    documented as such. Falls back to a unit affine when geometry is absent.
    """
    sx = sy = sz = 1.0
    if ds is not None:
        ps = getattr(ds, "PixelSpacing", None)
        if ps is not None and len(ps) >= 2:
            sy, sx = float(ps[0]), float(ps[1])
        sz = float(
            getattr(ds, "SpacingBetweenSlices", None)
            or getattr(ds, "SliceThickness", None)
            or 1.0
        )
    affine = np.diag([-sx, -sy, sz, 1.0]).astype(np.float32)
    if ds is not None:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            affine[0, 3] = -float(ipp[0])
            affine[1, 3] = -float(ipp[1])
            affine[2, 3] = float(ipp[2])
    return affine


def save_nifti(
    sample: Sample,
    path: str | Path,
    affine: Optional[np.ndarray] = None,
) -> Path:
    """Write the Sample's image to a single-file NIfTI-1 (.nii).

    affine: 4x4 voxel->world matrix. If None, derived from the Sample's
        source DICOM geometry (or a unit affine if none is available).
    """
    path = Path(path)
    vol_zyx = _single_volume(sample.image)  # (Z, Y, X)

    # NIfTI indexes [i, j, k] = [x, y, z] with i (x) fastest on disk. Move to
    # (X, Y, Z) and write in Fortran order so x varies fastest.
    vol_xyz = np.ascontiguousarray(np.transpose(vol_zyx, (2, 1, 0)), dtype=np.float32)
    nx, ny, nz = vol_xyz.shape

    if affine is None:
        affine = _affine_from_dataset(sample.source_dataset)
    affine = np.asarray(affine, dtype=np.float32)

    sx = float(np.linalg.norm(affine[:3, 0])) or 1.0
    sy = float(np.linalg.norm(affine[:3, 1])) or 1.0
    sz = float(np.linalg.norm(affine[:3, 2])) or 1.0

    hdr = bytearray(_NIFTI1_HEADER_SIZE)

    struct.pack_into("<i", hdr, 0, _NIFTI1_HEADER_SIZE)          # sizeof_hdr
    # dim[8]
    struct.pack_into("<8h", hdr, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    struct.pack_into("<h", hdr, 70, _DT_FLOAT32)                 # datatype
    struct.pack_into("<h", hdr, 72, 32)                          # bitpix
    # pixdim[8]: pixdim[0]=qfac (unused with sform), then voxel sizes
    struct.pack_into("<8f", hdr, 76, 1.0, sx, sy, sz, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", hdr, 108, _VOX_OFFSET)               # vox_offset
    struct.pack_into("<f", hdr, 112, 1.0)                       # scl_slope
    struct.pack_into("<f", hdr, 116, 0.0)                       # scl_inter
    struct.pack_into("<h", hdr, 252, 0)                         # qform_code
    struct.pack_into("<h", hdr, 254, 1)                         # sform_code = scanner
    # srow_x, srow_y, srow_z  (affine rows)
    struct.pack_into("<4f", hdr, 280, *[float(v) for v in affine[0]])
    struct.pack_into("<4f", hdr, 296, *[float(v) for v in affine[1]])
    struct.pack_into("<4f", hdr, 312, *[float(v) for v in affine[2]])
    struct.pack_into("<4s", hdr, 344, b"n+1\x00")              # magic

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(hdr))
        f.write(b"\x00\x00\x00\x00")  # pad 348 -> 352 (vox_offset)
        f.write(vol_xyz.tobytes(order="F"))
    return path


def dicom_to_nifti(
    in_path: str | Path,
    out_path: str | Path,
    normalize: Optional["object"] = None,
) -> Path:
    """Convenience: read a DICOM, optionally normalize, write a NIfTI.

    normalize: an IntensityNormalize (or any Transform) to apply before
        writing, or None to export raw intensities.
    """
    from .dicom_io import load_sample

    sample = load_sample(in_path)
    if normalize is not None:
        sample = normalize(sample)
    return save_nifti(sample, out_path)
