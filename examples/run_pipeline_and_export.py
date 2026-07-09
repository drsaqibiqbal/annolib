"""End-to-end example: run a pipeline on synthetic data, export a review bundle.

python examples/run_pipeline_and_export.py
streamlit run dashboard/review_app.py -- --bundle ./example_bundle
"""

from __future__ import annotations

import numpy as np

from prostate_deid import Compose, DicomTagScrub, OrganMask, Pseudonymize, Sample
from prostate_deid.export import export_review_bundle
from prostate_deid.organ_mask import WHOLE_GLAND

if __name__ == "__main__":
    shape = (1, 8, 64, 64)
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 255, size=shape).astype(np.float32)

    mask = np.zeros(shape[1:], dtype=np.uint8)
    mask[:, 16:48, 16:48] = 3

    sample = Sample(
        image=image,
        dicom_meta={
            "PatientName": "Doe^John",
            "PatientID": "SYNTH-0001",
            "StudyDate": "20260101",
        },
        mask=mask,
    )

    pipeline = Compose(
        [
            Pseudonymize(mapping_store="./example_mapping.json", always_apply=True),
            DicomTagScrub(
                profile="hipaa_safe_harbor",
                exclude_tags=["PatientID", "StudyDate"],
                always_apply=True,
            ),
            OrganMask(keep_labels=WHOLE_GLAND, always_apply=True),
        ]
    )

    result = pipeline(sample)

    for entry in result.audit:
        print(entry)

    bundle_path = export_review_bundle(result, "./example_bundle")
    print(f"\nExported review bundle to {bundle_path}")
