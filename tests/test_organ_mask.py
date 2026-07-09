import numpy as np

from prostate_deid import OrganMask, Sample
from prostate_deid.organ_mask import LESION, PERIPHERAL_ZONE, TRANSITION_ZONE, WHOLE_GLAND


def test_keeps_only_whole_gland_by_default(synthetic_prostate_volume):
    image, mask = synthetic_prostate_volume
    sample = Sample(image=image.copy(), mask=mask)

    transform = OrganMask(always_apply=True)
    result = transform(sample)

    outside = mask == 0
    assert np.all(result.image[:, outside] == 0)
    inside = mask != 0
    assert np.any(result.image[:, inside] != 0)


def test_keep_labels_lesion_only(synthetic_prostate_volume):
    image, mask = synthetic_prostate_volume
    sample = Sample(image=image.copy(), mask=mask)

    transform = OrganMask(keep_labels=(LESION,), always_apply=True)
    result = transform(sample)

    non_lesion = mask != LESION
    assert np.all(result.image[:, non_lesion] == 0)


def test_keep_labels_both_zones(synthetic_prostate_volume):
    image, mask = synthetic_prostate_volume
    sample = Sample(image=image.copy(), mask=mask)

    transform = OrganMask(keep_labels=(PERIPHERAL_ZONE, TRANSITION_ZONE), always_apply=True)
    result = transform(sample)

    kept = np.isin(mask, [PERIPHERAL_ZONE, TRANSITION_ZONE])
    assert np.all(result.image[:, ~kept] == 0)


def test_no_mask_is_noop():
    sample = Sample(image=np.ones((1, 4, 4, 4)), mask=None)
    transform = OrganMask(always_apply=True)

    result = transform(sample)

    assert np.all(result.image == 1)
    assert result.audit == []


def test_custom_fill_value(synthetic_prostate_volume):
    image, mask = synthetic_prostate_volume
    sample = Sample(image=image.copy(), mask=mask)

    transform = OrganMask(fill_value=-1.0, always_apply=True)
    result = transform(sample)

    outside = mask == 0
    assert np.all(result.image[:, outside] == -1.0)


def test_logs_audit_entry_with_voxel_count(synthetic_prostate_volume):
    image, mask = synthetic_prostate_volume
    sample = Sample(image=image.copy(), mask=mask)

    transform = OrganMask(keep_labels=WHOLE_GLAND, always_apply=True)
    result = transform(sample)

    assert len(result.audit) == 1
    assert result.audit[0].step == "organ_mask"
    assert "voxels_zeroed=" in result.audit[0].detail
