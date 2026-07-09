# prostate-deid

A Python library for de-identifying and anatomically masking prostate MRI
studies (DICOM), producing a single auditable report of every change made.

It composes four operations behind one `Transform` / `Compose` API — the
pattern established by volumetric augmentation libraries such as
Bio-Volumentations:

| Transform | Purpose |
|---|---|
| `DicomTagScrub` | Deterministic DICOM header scrubbing (HIPAA Safe Harbor / DICOM PS3.15), private tags stripped by default |
| `Pseudonymize` | Consistent per-patient ID remap + date shift, mapping key stored separately |
| `BurnedInTextRedact` | OCR + NER detection of PHI burned into pixel data or DICOM overlay planes |
| `OrganMask` | Prostate zonal / lesion masking (peripheral zone, transition zone, lesion) |

Every transform appends a structured `AuditEntry` to the sample instead of
mutating data silently, so a pipeline run produces one machine-readable
compliance report alongside the de-identified output.

See [Quickstart](quickstart.md) to get started, or
[Anonymization vs. Pseudonymization](pseudonymization.md) for the design
decision behind `Pseudonymize` before you use it on real data.

!!! warning "Compliance disclaimer"
    Using this library does not, by itself, constitute HIPAA or GDPR
    compliance. It is a tool toward compliance, not a certification of it.
    See the [README](https://github.com/<user>/prostate-deid#compliance-disclaimer)
    for the full disclaimer.
