"""Synthetic fixtures only -- no real patient data in this repo, ever.

Per the project's critical_considerations doc: test fixtures must be
synthetically generated PHI (fake names, fake dates, synthetic burned-in
text), not real "de-identified" datasets whose de-identification claim was
someone else's. This mirrors the design principle behind TCIA's
Pseudo-PHI-DICOM-Data collection without depending on that external dataset
in CI.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_dicom_meta():
    return {
        "PatientName": "Doe^John",
        "PatientID": "SYNTH-0001",
        "PatientBirthDate": "19700101",
        "StudyDate": "20260101",
        "SeriesDate": "20260101",
        "ReferringPhysicianName": "Smith^Jane",
        "InstitutionName": "Synthetic General Hospital",
        "StudyID": "1",
        "AccessionNumber": "ACC123",
        "Manufacturer": "SyntheticVendor",
        "0x00410010": "vendor private block",
    }


@pytest.fixture
def synthetic_prostate_volume():
    """A small synthetic volume with a coarse gland/zone/lesion mask.

    Shapes are tiny on purpose -- this is a unit-test fixture, not a
    realistic-resolution phantom.
    """
    shape = (1, 32, 32, 32)  # C, Z, Y, X
    rng = np.random.default_rng(seed=42)
    image = rng.uniform(0, 255, size=shape).astype(np.float32)

    mask = np.zeros(shape[1:], dtype=np.uint8)
    center = np.array(shape[1:]) // 2
    zz, yy, xx = np.ogrid[: shape[1], : shape[2], : shape[3]]
    dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    mask[dist < 6] = 3  # transition zone (inner)
    mask[(dist >= 6) & (dist < 10)] = 2  # peripheral zone (outer ring)
    mask[(zz == center[0]) & (yy == center[1]) & (xx == center[2])] = 4  # lesion voxel

    return image, mask
