"""One-call anonymization entry point.

`anonymize_dicom(in_path, out_path)` is the highest-level API: read a real
.dcm, run a sensible default de-identification pipeline (pseudonymize IDs and
dates, regenerate UIDs, scrub the header per HIPAA Safe Harbor, strip private
tags, best-effort burned-in text redaction), write a de-identified .dcm, and
return the audit trail.

This is deliberately opinionated so the common case is one call, while the
Compose API remains available for users who need a custom pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .burned_in_text import BurnedInTextRedact
from .core import AuditEntry, Compose
from .dicom_io import load_sample, save_sample
from .dicom_scrub import DicomTagScrub
from .pseudonymize import Pseudonymize

# Fields Pseudonymize remaps (rather than deletes) must be excluded from the
# scrub profile, or the scrub would delete the pseudonymized values.
_PSEUDONYMIZED_FIELDS = [
    "PatientID",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
]


def default_pipeline(
    mapping_store: "str | Path",
    profile: str = "hipaa_safe_harbor",
    redact_burned_in_text: bool = True,
) -> Compose:
    """The library's recommended default de-identification pipeline."""
    steps: list = [
        Pseudonymize(mapping_store=mapping_store, always_apply=True),
        DicomTagScrub(
            profile=profile,
            exclude_tags=_PSEUDONYMIZED_FIELDS,
            strip_private_tags=True,
            always_apply=True,
        ),
    ]
    if redact_burned_in_text:
        # require_ocr=False: if tesseract is absent, this records a flagged
        # audit entry rather than crashing the whole run.
        steps.append(BurnedInTextRedact(require_ocr=False, always_apply=True))
    return Compose(steps)


def anonymize_dicom(
    in_path: "str | Path",
    out_path: "str | Path",
    mapping_store: "str | Path" = "./prostate_deid_mapping.json",
    profile: str = "hipaa_safe_harbor",
    redact_burned_in_text: bool = True,
) -> List[AuditEntry]:
    """Read a .dcm, de-identify it, write a .dcm, and return the audit trail."""
    sample = load_sample(in_path)
    pipeline = default_pipeline(
        mapping_store=mapping_store,
        profile=profile,
        redact_burned_in_text=redact_burned_in_text,
    )
    result = pipeline(sample)
    save_sample(result, out_path)
    return result.audit


def summarize_audit(audit: List[AuditEntry]) -> dict:
    """Aggregate an audit trail into headline counts for reporting/dashboards."""
    summary = {
        "total_actions": len(audit),
        "tags_removed": 0,
        "private_tags_removed": 0,
        "ids_pseudonymized": 0,
        "dates_shifted": 0,
        "uids_regenerated": 0,
        "pixel_regions_redacted": 0,
        "flagged_for_review": 0,
    }
    for e in audit:
        if e.action == "removed_tag":
            summary["tags_removed"] += 1
        elif e.action == "removed_private_tag":
            summary["private_tags_removed"] += 1
        elif e.action == "remapped_id":
            summary["ids_pseudonymized"] += 1
        elif e.action == "shifted_date":
            summary["dates_shifted"] += 1
        elif e.action == "regenerated_uid":
            summary["uids_regenerated"] += 1
        elif e.action in ("redacted_region", "redacted_overlay_region"):
            summary["pixel_regions_redacted"] += 1
        if e.flagged_for_review:
            summary["flagged_for_review"] += 1
    return summary
