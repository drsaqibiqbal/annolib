import json

import numpy as np

from prostate_deid import AuditEntry, Sample
from prostate_deid.export import export_review_bundle


def test_export_writes_audit_json(tmp_path):
    sample = Sample(image=np.zeros((32, 32), dtype=np.float32))
    sample.log(AuditEntry(step="dicom_tag_scrub", action="removed_tag", detail="PatientName"))

    out_dir = tmp_path / "bundle"
    audit_path = export_review_bundle(sample, out_dir)

    assert audit_path.exists()
    rows = json.loads(audit_path.read_text())
    assert len(rows) == 1
    assert rows[0]["step"] == "dicom_tag_scrub"
    assert rows[0]["thumbnail"] is None


def test_export_writes_thumbnail_for_flagged_bbox_entry(tmp_path):
    image = np.zeros((32, 32), dtype=np.float32)
    image[4:8, 4:8] = 99
    sample = Sample(image=image)
    sample.log(
        AuditEntry(
            step="burned_in_phi",
            action="redacted_region",
            detail="bbox=(4, 8, 4, 8), text_hash=123",
            confidence=0.3,
            flagged_for_review=True,
        )
    )

    out_dir = tmp_path / "bundle"
    audit_path = export_review_bundle(sample, out_dir)

    rows = json.loads(audit_path.read_text())
    assert rows[0]["thumbnail"] is not None

    thumb_path = out_dir / rows[0]["thumbnail"]
    assert thumb_path.exists()
    crop = np.load(thumb_path)
    assert crop.shape == (4, 4)


def test_non_flagged_entries_have_no_thumbnail(tmp_path):
    sample = Sample(image=np.zeros((32, 32), dtype=np.float32))
    sample.log(
        AuditEntry(
            step="burned_in_phi",
            action="redacted_region",
            detail="bbox=(0, 4, 0, 4), text_hash=1",
            confidence=0.95,
            flagged_for_review=False,
        )
    )

    out_dir = tmp_path / "bundle"
    audit_path = export_review_bundle(sample, out_dir)

    rows = json.loads(audit_path.read_text())
    assert rows[0]["thumbnail"] is None
