# Reviewer QA Dashboard

`flagged_for_review=True` on an `AuditEntry` is the right instinct but, on
its own, a dead-end data point with no consumer. This dashboard is that
consumer: it loads an exported audit bundle and surfaces every flagged
region as a cropped thumbnail for quick visual triage.

## Usage

1. Export a bundle after running a pipeline:

    ```python
    from prostate_deid.export import export_review_bundle
    export_review_bundle(result, "./review_bundle")
    ```

2. Launch the dashboard against it:

    ```bash
    pip install prostate-deid[dashboard]
    streamlit run dashboard/review_app.py -- --bundle ./review_bundle
    ```

3. For each flagged entry, the dashboard shows the cropped image region (when
   available), the detection confidence, and a disposition control
   (confirmed correct / false positive). Entries without a bounding box
   (e.g. `Pseudonymize` failures) are still listed, without a thumbnail.

## Why this exists

Per the project's design principles: false negatives (missed PHI) and false
positives (over-redaction) are not symmetric failures — a missed detection
is a privacy incident, an over-redaction is a data-quality loss. Defaults
across the library bias toward flagging under uncertainty. That default only
pays off if a human actually looks at what got flagged — otherwise "flagged
for review" quietly becomes "flagged and ignored," which creates false
confidence and is worse than not flagging at all.
