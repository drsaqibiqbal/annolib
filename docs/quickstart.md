# Quickstart

## Install

```bash
pip install prostate-deid
pip install prostate-deid[ocr]        # for BurnedInTextRedact
pip install prostate-deid[dashboard]  # for the reviewer QA dashboard
```

## Build a pipeline

```python
from prostate_deid import Compose, DicomTagScrub, Pseudonymize, OrganMask, Sample
from prostate_deid.organ_mask import WHOLE_GLAND

pipeline = Compose([
    # Pseudonymize must run before DicomTagScrub -- see the ordering note
    # on DicomTagScrub for why.
    Pseudonymize(mapping_store="/secure/institution-only/mapping.json", always_apply=True),
    DicomTagScrub(
        profile="hipaa_safe_harbor",
        exclude_tags=["PatientID", "StudyDate", "SeriesDate"],
        always_apply=True,
    ),
    OrganMask(keep_labels=WHOLE_GLAND, always_apply=True),
])

sample = Sample(image=image_array, dicom_meta=dicom_header_dict, mask=segmentation_mask)
result = pipeline(sample)

for entry in result.audit:
    print(entry)
```

## Export a review bundle

```python
from prostate_deid.export import export_review_bundle

export_review_bundle(result, "./review_bundle")
```

Then launch the dashboard:

```bash
streamlit run dashboard/review_app.py -- --bundle ./review_bundle
```

See [`examples/run_pipeline_and_export.py`](https://github.com/drsaqibiqbal/annolib/blob/main/examples/run_pipeline_and_export.py)
for a runnable end-to-end script on synthetic data.
