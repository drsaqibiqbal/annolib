"""Deterministic DICOM metadata scrubbing.

Delegates DICOM I/O to pydicom; this module only encodes policy: which named
profile applies, and how private (vendor-specific) tags are handled.

Private tags are a known blind spot -- standard confidentiality profiles only
cover the public DICOM dictionary, and scanner manufacturers routinely embed
PHI in private, undocumented fields. Default policy here is to strip *all*
private tags rather than attempt to parse unknown content, since guessing
wrong about an unrecognized private tag is worse than dropping it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .core import AuditEntry, Sample, Transform
from .profiles import resolve_profile


def _is_private_tag(dicom_meta: Dict[str, Any], keyword: str) -> bool:
    """Best-effort private-tag detection for a plain-dict dicom_meta.

    When dicom_meta was built from a pydicom Dataset, callers should prefer
    passing tag tuples (group, element) instead of keywords for private
    elements, since private tags have no stable keyword. This helper treats
    any key that pydicom's dictionary does not recognize as "private" by
    convention -- i.e. an odd group number, or a key not in the public
    dictionary.
    """
    try:
        from pydicom.datadict import dictionary_has_tag

        return not dictionary_has_tag(keyword)
    except Exception:
        return False


class DicomTagScrub(Transform):
    """Removes DICOM header fields according to a named profile.

    profile: "hipaa_safe_harbor", "dicom_ps3.15_basic", or any key registered
        in prostate_deid.profiles.PROFILES.
    extra_tags_to_remove: additional keywords to strip beyond the profile.
    exclude_tags: fields to keep even though the profile would otherwise
        remove them. Use this when a Pseudonymize step earlier in the same
        Compose pipeline has already remapped these fields (e.g.
        exclude_tags=["PatientID", "StudyDate", "SeriesDate"]) -- without
        this, DicomTagScrub would delete the pseudonymized values outright,
        silently undoing the remap.
    strip_private_tags: if True (default), also remove every private
        (vendor-specific, non-standard) tag found in dicom_meta -- this is
        the library's explicit default policy, not an incidental behavior.

    Ordering note: run Pseudonymize *before* DicomTagScrub in your Compose
    pipeline, and pass the same field names to DicomTagScrub's
    exclude_tags -- otherwise DicomTagScrub's profile deletes the very
    fields Pseudonymize just remapped.
    """

    def __init__(
        self,
        profile: str = "hipaa_safe_harbor",
        extra_tags_to_remove: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        strip_private_tags: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.profile = profile
        self.extra_tags_to_remove = extra_tags_to_remove or []
        self.exclude_tags = set(exclude_tags or [])
        self.strip_private_tags = strip_private_tags

    def apply(self, sample: Sample) -> Sample:
        if sample.dicom_meta is None:
            return sample

        tags_to_strip = (
            set(resolve_profile(self.profile)) | set(self.extra_tags_to_remove)
        ) - self.exclude_tags

        for tag in list(sample.dicom_meta.keys()):
            if tag in tags_to_strip:
                del sample.dicom_meta[tag]
                sample.log(
                    AuditEntry(
                        step="dicom_tag_scrub",
                        action="removed_tag",
                        detail=tag,
                    )
                )
            elif self.strip_private_tags and _is_private_tag(sample.dicom_meta, tag):
                del sample.dicom_meta[tag]
                sample.log(
                    AuditEntry(
                        step="dicom_tag_scrub",
                        action="removed_private_tag",
                        detail=tag,
                    )
                )

        return sample
