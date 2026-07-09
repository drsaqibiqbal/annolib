"""End-to-end pipeline test.

Runs a full Compose chain on a synthetic sample so refactors that break the
chaining/audit-accumulation logic itself get caught, not just individual
transform bugs.
"""

import numpy as np

from prostate_deid import Compose, DicomTagScrub, OrganMask, Pseudonymize, Sample
from prostate_deid.organ_mask import WHOLE_GLAND


def test_full_pipeline_accumulates_one_shared_audit_trail(
    tmp_path, synthetic_dicom_meta, synthetic_prostate_volume
):
    image, mask = synthetic_prostate_volume
    sample = Sample(image=image.copy(), dicom_meta=dict(synthetic_dicom_meta), mask=mask)

    pipeline = Compose(
        [
            # Pseudonymize must run before DicomTagScrub, and DicomTagScrub
            # must exclude the fields Pseudonymize just remapped -- otherwise
            # the HIPAA profile deletes the pseudonymized values outright.
            Pseudonymize(mapping_store=tmp_path / "mapping.json", always_apply=True),
            DicomTagScrub(
                profile="hipaa_safe_harbor",
                exclude_tags=["PatientID", "StudyDate", "SeriesDate"],
                always_apply=True,
            ),
            OrganMask(keep_labels=WHOLE_GLAND, always_apply=True),
        ]
    )

    result = pipeline(sample)

    assert "PatientName" not in result.dicom_meta
    assert result.dicom_meta["PatientID"] != "SYNTH-0001"
    assert np.all(result.image[:, mask == 0] == 0)

    steps = {e.step for e in result.audit}
    assert steps == {"dicom_tag_scrub", "pseudonymize", "organ_mask"}
