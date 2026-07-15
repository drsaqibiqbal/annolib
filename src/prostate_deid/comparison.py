"""Capability comparison against related de-identification tools.

IMPORTANT — what this is and is not:

This is a *capability* (feature-support) comparison, not a quantitative
performance benchmark. Each cell states whether a tool addresses a capability
at all, based on that tool's public documentation. It does NOT claim a
detection-recall or accuracy number for any tool, including this one.

Quantitative claims ("X% PHI-detection recall vs. tool Y") require running
all tools on a shared, ground-truthed evaluation set (e.g. the TCIA
Pseudo-PHI-DICOM-Data collection or the MIDI-B challenge data) and are
deliberately absent until that validation is done. Fabricating such numbers
would be scientific misconduct for a tool of this kind; see
critical_considerations: "validate against MIDI-B, not your own held-out set."

Verify each cell against the current state of each tool before citing this
table in a publication.
"""

from __future__ import annotations

from typing import Dict, List

# Capabilities compared, in display order.
CAPABILITIES: List[str] = [
    "DICOM metadata scrubbing (profile-based)",
    "Private (vendor) tag stripping",
    "Consistent ID pseudonymization",
    "Consistent date shifting",
    "Consistent UID regeneration",
    "Burned-in pixel text redaction (OCR+NER)",
    "Overlay-plane scrubbing",
    "Prostate zonal / lesion masking",
    "NIfTI export + intensity normalization",
    "Per-action confidence scoring",
    "Unified audit log across all steps",
    "Human-review dashboard for flagged items",
    "Single Compose-style pipeline API",
]

# Support level per tool per capability: "full", "partial", or "none".
# Sourced from each tool's public documentation; re-verify before publishing.
TOOLS: Dict[str, Dict[str, str]] = {
    "Presidio Image Redactor": {
        "DICOM metadata scrubbing (profile-based)": "none",
        "Private (vendor) tag stripping": "none",
        "Consistent ID pseudonymization": "none",
        "Consistent date shifting": "none",
        "Consistent UID regeneration": "none",
        "Burned-in pixel text redaction (OCR+NER)": "full",
        "Overlay-plane scrubbing": "none",
        "Prostate zonal / lesion masking": "none",
        "NIfTI export + intensity normalization": "none",
        "Per-action confidence scoring": "partial",
        "Unified audit log across all steps": "none",
        "Human-review dashboard for flagged items": "none",
        "Single Compose-style pipeline API": "none",
    },
    "deid (Vanderbilt / pydicom)": {
        "DICOM metadata scrubbing (profile-based)": "full",
        "Private (vendor) tag stripping": "partial",
        "Consistent ID pseudonymization": "partial",
        "Consistent date shifting": "partial",
        "Consistent UID regeneration": "partial",
        "Burned-in pixel text redaction (OCR+NER)": "none",
        "Overlay-plane scrubbing": "none",
        "Prostate zonal / lesion masking": "none",
        "NIfTI export + intensity normalization": "none",
        "Per-action confidence scoring": "none",
        "Unified audit log across all steps": "partial",
        "Human-review dashboard for flagged items": "none",
        "Single Compose-style pipeline API": "none",
    },
    "CTP DicomAnonymizer": {
        "DICOM metadata scrubbing (profile-based)": "full",
        "Private (vendor) tag stripping": "full",
        "Consistent ID pseudonymization": "full",
        "Consistent date shifting": "full",
        "Consistent UID regeneration": "full",
        "Burned-in pixel text redaction (OCR+NER)": "none",
        "Overlay-plane scrubbing": "partial",
        "Prostate zonal / lesion masking": "none",
        "NIfTI export + intensity normalization": "none",
        "Per-action confidence scoring": "none",
        "Unified audit log across all steps": "partial",
        "Human-review dashboard for flagged items": "none",
        "Single Compose-style pipeline API": "none",
    },
    "prostate-deid (this library)": {
        "DICOM metadata scrubbing (profile-based)": "full",
        "Private (vendor) tag stripping": "full",
        "Consistent ID pseudonymization": "full",
        "Consistent date shifting": "full",
        "Consistent UID regeneration": "full",
        "Burned-in pixel text redaction (OCR+NER)": "full",
        "Overlay-plane scrubbing": "full",
        "Prostate zonal / lesion masking": "full",
        "NIfTI export + intensity normalization": "full",
        "Per-action confidence scoring": "full",
        "Unified audit log across all steps": "full",
        "Human-review dashboard for flagged items": "full",
        "Single Compose-style pipeline API": "full",
    },
}

_SYMBOL = {"full": "✓", "partial": "~", "none": "–"}


def as_rows() -> List[Dict[str, str]]:
    """Return the comparison as a list of row dicts for tabular display."""
    rows = []
    for cap in CAPABILITIES:
        row = {"Capability": cap}
        for tool, support in TOOLS.items():
            row[tool] = _SYMBOL[support.get(cap, "none")]
        rows.append(row)
    return rows


def as_markdown() -> str:
    """Return the comparison table as GitHub-flavoured Markdown."""
    tools = list(TOOLS.keys())
    header = "| Capability | " + " | ".join(tools) + " |"
    sep = "|" + "---|" * (len(tools) + 1)
    lines = [header, sep]
    for row in as_rows():
        cells = [row["Capability"]] + [row[t] for t in tools]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Legend: ✓ full support &nbsp; ~ partial &nbsp; – not addressed. "
                 "Capability comparison only — not a quantitative performance benchmark.")
    return "\n".join(lines)
