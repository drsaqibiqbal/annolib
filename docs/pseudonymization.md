# Anonymization vs. Pseudonymization

**This library performs pseudonymization, not anonymization.**

Real de-identification pipelines usually don't simply delete identifiers —
longitudinal research (PSA follow-up, repeat MRI across a treatment course)
needs to know that scan A and scan B belong to the same patient at different
time points, and date deltas (e.g. time since diagnosis) often carry real
research value.

`Pseudonymize` implements this by:

- Deriving a **consistent** pseudo patient ID per source `PatientID`, via
  HMAC-SHA256 keyed on a secret you control (`PROSTATE_DEID_KEY` or
  `secret=`).
- Deriving a **consistent per-patient date-shift offset** and applying it to
  every configured date field, so relative timing between a patient's scans
  is preserved even though absolute dates are not.
- Persisting the patient ID → (pseudo ID, shift) mapping to a
  `MappingStore` (a local JSON file by default) that **you must place
  outside** the directory tree containing the de-identified images.

## Why this matters legally

Under GDPR, retaining *any* mapping that could re-identify a patient — even
one held only by the originating institution, even encrypted — makes the
processed output **pseudonymized data**, not **anonymized data** (Art. 4(5)).
These are legally distinct categories with different obligations. This
library does not claim, and cannot certify, that its output is anonymized.

## If you need true anonymization instead

Don't use `Pseudonymize`. Irreversibly discard `PatientID` and date fields
(e.g. via `DicomTagScrub`'s HIPAA Safe Harbor profile, which deletes them
outright) and do not retain any mapping. Note this breaks cross-scan
linkage for the same patient — a real tradeoff, not a bug.

## Key management is your responsibility

- Set `PROSTATE_DEID_KEY` to a real secret before running in production —
  the default is empty and explicitly unsuitable for anything but tests.
- The mapping store file is as sensitive as a plaintext patient list. Treat
  its storage location, access control, and backup policy accordingly.
- Never commit a mapping store file to a repository, or place it alongside
  the de-identified images it maps.
