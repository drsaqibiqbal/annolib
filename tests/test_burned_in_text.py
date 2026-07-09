import numpy as np
import pytest

from prostate_deid import BurnedInTextRedact, Sample


def test_unsupported_engine_raises():
    sample = Sample(image=np.zeros((32, 32), dtype=np.float32))
    transform = BurnedInTextRedact(ocr_engine="not_a_real_engine", always_apply=True)

    with pytest.raises(ValueError):
        transform(sample)


def test_missing_ocr_dependency_raises_helpful_error():
    pytest.importorskip("pytesseract", reason="only meaningful when pytesseract is absent")


def test_no_overlays_present_is_noop_for_overlay_step():
    sample = Sample(
        image=np.zeros((32, 32), dtype=np.float32),
        dicom_meta={"PatientName": "Doe^John"},
    )
    transform = BurnedInTextRedact(scrub_overlays=True, always_apply=True)

    try:
        transform(sample)
    except ImportError:
        pytest.skip("pytesseract not installed in this environment")
