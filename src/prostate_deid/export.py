"""Serialize a Sample's audit trail + flagged regions to disk for review.

The dashboard reads this format rather than a live Sample object, so a
pipeline run (which may happen on a headless server, in CI, or on a machine
without a display) can be reviewed later or elsewhere.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .core import Sample

_BBOX_RE = re.compile(r"bbox=\((\d+), (\d+), (\d+), (\d+)\)")


def export_review_bundle(sample: Sample, out_dir: str | Path) -> Path:
    """Writes audit.json + one .npy thumbnail per flagged, bbox-bearing entry.

    out_dir layout:
        out_dir/audit.json       -- full audit trail, one row per entry
        out_dir/thumbnails/NNN.npy  -- cropped region for flagged entries that
                                       carry a bbox in their `detail` string
    """
    out_dir = Path(out_dir)
    thumb_dir = out_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, entry in enumerate(sample.audit):
        row = {
            "index": i,
            "step": entry.step,
            "action": entry.action,
            "detail": entry.detail,
            "confidence": entry.confidence,
            "flagged_for_review": entry.flagged_for_review,
            "thumbnail": None,
        }

        if entry.flagged_for_review:
            match = _BBOX_RE.search(entry.detail)
            if match:
                y0, y1, x0, x1 = (int(g) for g in match.groups())
                frame = sample.image if sample.image.ndim == 2 else sample.image[0]
                crop = np.asarray(frame[..., y0:y1, x0:x1])
                thumb_path = thumb_dir / f"{i:04d}.npy"
                np.save(thumb_path, crop)
                row["thumbnail"] = str(thumb_path.relative_to(out_dir))

        rows.append(row)

    audit_path = out_dir / "audit.json"
    audit_path.write_text(json.dumps(rows, indent=2))
    return audit_path
