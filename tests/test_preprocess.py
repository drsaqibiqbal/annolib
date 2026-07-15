"""Tests for the preprocessing additions: NIfTI export and RARN normalization.

The NIfTI writer is self-contained (numpy only); its output is validated here
against nibabel as the reference reader, so we know the header/affine/data are
well-formed even though the library itself has no nibabel dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from prostate_deid import IntensityNormalize, Sample, save_nifti
from prostate_deid.organ_mask import PERIPHERAL_ZONE, TRANSITION_ZONE, WHOLE_GLAND


# ----------------------------- NIfTI export --------------------------------
def test_nifti_roundtrip_matches_nibabel(tmp_path):
    nib = pytest.importorskip("nibabel")
    rng = np.random.default_rng(0)
    vol_zyx = rng.uniform(0, 500, size=(6, 8, 10)).astype(np.float32)  # Z,Y,X
    sample = Sample(image=vol_zyx[np.newaxis, ...])  # C,Z,Y,X

    out = save_nifti(sample, tmp_path / "vol.nii")
    assert out.exists()

    img = nib.load(str(out))
    data = np.asarray(img.dataobj)
    # nibabel returns [x, y, z]; our source is [z, y, x] -> transpose to compare.
    assert data.shape == (10, 8, 6)
    np.testing.assert_allclose(
        data, np.transpose(vol_zyx, (2, 1, 0)), rtol=1e-5, atol=1e-4
    )


def test_nifti_affine_encodes_spacing(tmp_path):
    nib = pytest.importorskip("nibabel")

    class FakeDS:
        PixelSpacing = [0.5, 0.75]  # [row(y), col(x)]
        SliceThickness = 3.0

    sample = Sample(image=np.zeros((1, 4, 4, 4), dtype=np.float32), source_dataset=FakeDS())
    out = save_nifti(sample, tmp_path / "vol.nii")

    img = nib.load(str(out))
    zooms = img.header.get_zooms()[:3]
    assert zooms[0] == pytest.approx(0.75)  # x spacing (col)
    assert zooms[1] == pytest.approx(0.5)   # y spacing (row)
    assert zooms[2] == pytest.approx(3.0)   # z spacing


# --------------------------- normalization ---------------------------------
def _synthetic_masked_sample(scale=1.0, offset=0.0):
    rng = np.random.default_rng(1)
    img = (rng.normal(100, 20, size=(1, 8, 16, 16)) * scale + offset).astype(np.float32)
    mask = np.zeros((8, 16, 16), dtype=np.uint8)
    mask[:, 4:8, 4:8] = TRANSITION_ZONE
    mask[:, 8:12, 8:12] = PERIPHERAL_ZONE
    return Sample(image=img, mask=mask)


def test_robust_normalization_centers_reference_near_zero():
    sample = _synthetic_masked_sample()
    ref_before = sample.image[:, sample.mask != 0]

    IntensityNormalize(method="robust", always_apply=True)(sample)

    ref_after = sample.image[:, sample.mask != 0]
    # Reference median should map close to 0 and robust scale near 1.
    assert abs(float(np.median(ref_after))) < 0.15
    assert 0.5 < np.std(ref_after) < 2.0
    assert float(np.median(ref_before)) != 0.0  # sanity: it moved


def test_normalization_is_scanner_invariant():
    """Two scans of the same content at different scanner gains should
    normalize to nearly the same distribution — the whole point of RARN."""
    a = _synthetic_masked_sample(scale=1.0, offset=0.0)
    b = _synthetic_masked_sample(scale=2.5, offset=40.0)  # different "scanner"

    IntensityNormalize(method="robust", always_apply=True)(a)
    IntensityNormalize(method="robust", always_apply=True)(b)

    ma = a.image[:, a.mask != 0]
    mb = b.image[:, b.mask != 0]
    assert abs(float(np.median(ma)) - float(np.median(mb))) < 0.1
    assert abs(float(np.std(ma)) - float(np.std(mb))) < 0.15


def test_minmax_maps_into_unit_range():
    sample = _synthetic_masked_sample()
    IntensityNormalize(method="minmax", always_apply=True)(sample)
    assert sample.image.min() >= -1e-6
    assert sample.image.max() <= 1.0 + 1e-6


def test_audit_records_reversible_affine():
    sample = _synthetic_masked_sample()
    IntensityNormalize(method="robust", always_apply=True)(sample)
    entries = [e for e in sample.audit if e.step == "intensity_normalize"]
    assert len(entries) == 1
    assert "mu=" in entries[0].detail and "sigma=" in entries[0].detail
    assert "reversible_affine" in entries[0].detail


def test_reversibility_on_unclipped_support():
    """I(x) = I~(x)*sigma + mu must recover the original when no clipping
    occurs. Use a whole-image reference at (0,100) percentiles so the clip
    bounds are the global min/max and nothing is truncated."""
    rng = np.random.default_rng(2)
    img = rng.normal(100, 20, size=(1, 8, 16, 16)).astype(np.float32)
    sample = Sample(image=img)  # no mask -> reference can be the whole image
    original = sample.image.copy()

    norm = IntensityNormalize(
        method="robust",
        clip_percentiles=(0, 100),
        use_otsu_foreground=False,  # reference = whole image, so no clipping
        always_apply=True,
    )
    norm(sample)

    detail = [e for e in sample.audit if e.step == "intensity_normalize"][0].detail
    mu = float(detail.split("mu=")[1].split(",")[0])
    sigma = float(detail.split("sigma=")[1].split(",")[0])
    recovered = sample.image * sigma + mu
    np.testing.assert_allclose(recovered, original, rtol=1e-4, atol=1e-2)


def test_zone_aware_standardizes_each_zone():
    sample = _synthetic_masked_sample()
    # Give the two zones different intensity offsets so zone-aware has work to do.
    sample.image[:, sample.mask == PERIPHERAL_ZONE] += 60.0
    IntensityNormalize(method="zone_aware", always_apply=True)(sample)

    pz = sample.image[:, sample.mask == PERIPHERAL_ZONE]
    tz = sample.image[:, sample.mask == TRANSITION_ZONE]
    # After zone-aware standardization both zones sit near median 0.
    assert abs(float(np.median(pz))) < 0.2
    assert abs(float(np.median(tz))) < 0.2


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        IntensityNormalize(method="bogus")


def test_normalization_composes_in_pipeline():
    from prostate_deid import Compose
    from prostate_deid.organ_mask import OrganMask

    sample = _synthetic_masked_sample()
    pipe = Compose([
        IntensityNormalize(method="robust", always_apply=True),
        OrganMask(keep_labels=WHOLE_GLAND, always_apply=True),
    ])
    result = pipe(sample)
    steps = [e.step for e in result.audit]
    assert "intensity_normalize" in steps
    assert "organ_mask" in steps
