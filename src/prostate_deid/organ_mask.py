"""Prostate-specific anatomical masking.

Constrains an image to a segmented region -- whole gland, zonal
(peripheral/transition zone), or a lesion ROI -- sharing the same
mask-handling convention as augmentation-oriented dual transforms (image and
mask are touched consistently), so masks produced here remain usable in a
downstream augmentation pipeline built on the same convention.

Mask label convention (uint8, single-channel, same spatial shape as image).
Zone/lesion labels are mutually exclusive; WHOLE_GLAND is not a separate
label painted in the mask but shorthand for "any prostate label" (i.e. the
union of PZ, TZ, and lesion), since real segmentations typically label zones
exclusively rather than also painting a redundant outer "gland" label:
    0            = outside the gland (background)
    2            = peripheral zone (PZ)
    3            = transition zone (TZ)
    4            = lesion / suspicious ROI
Any subset of these labels may be present; labels not passed to `keep_labels`
are treated as "outside" and zeroed along with label 0.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .core import AuditEntry, Sample, Transform

PERIPHERAL_ZONE = 2
TRANSITION_ZONE = 3
LESION = 4
WHOLE_GLAND = (PERIPHERAL_ZONE, TRANSITION_ZONE, LESION)

_LABEL_NAMES = {
    PERIPHERAL_ZONE: "peripheral_zone",
    TRANSITION_ZONE: "transition_zone",
    LESION: "lesion",
}


class OrganMask(Transform):
    """Zeroes out image voxels outside the requested prostate region(s).

    keep_labels: which mask label(s) to retain, e.g. WHOLE_GLAND (the
        default) to constrain to the entire gland (PZ + TZ + lesion), or
        (PERIPHERAL_ZONE, TRANSITION_ZONE) to keep both zones but exclude
        the lesion label specifically.
    fill_value: value written into voxels outside the kept region(s).
    """

    def __init__(
        self,
        keep_labels: Iterable[int] = WHOLE_GLAND,
        fill_value: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.keep_labels = set(keep_labels)
        self.fill_value = fill_value

    def apply(self, sample: Sample) -> Sample:
        if sample.mask is None:
            return sample

        keep = np.isin(sample.mask, list(self.keep_labels))
        outside = ~keep

        sample.image = np.where(outside, self.fill_value, sample.image)
        sample.image_dirty = True

        label_desc = ",".join(
            _LABEL_NAMES.get(label, str(label)) for label in sorted(self.keep_labels)
        )
        sample.log(
            AuditEntry(
                step="organ_mask",
                action="masked_outside_region",
                detail=f"kept={label_desc}, voxels_zeroed={int(outside.sum())}",
            )
        )
        return sample
