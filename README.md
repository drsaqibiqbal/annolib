# prostate-deid

A Python library for de-identifying and anatomically masking prostate MRI
studies (DICOM), producing a single auditable report of every change made.

Composes four operations behind one `Transform` / `Compose` API (the pattern
established by volumetric augmentation libraries such as Bio-Volumentations):

- **`DicomTagScrub`** — deterministic DICOM header scrubbing (HIPAA Safe
  Harbor or DICOM PS3.15 profiles), with all unrecognized private
  (vendor-specific) tags stripped by default.
- **`Pseudonymize`** — consistent per-patient ID remapping and date shifting,
  so longitudinal studies (e.g. repeat MRI across a treatment course) stay
  linkable. See [Anonymization vs. pseudonymization](#anonymization-vs-pseudonymization) below —
  this is a legally meaningful distinction, not an implementation detail.
- **`BurnedInTextRedact`** — OCR + NER detection of PHI burned into pixel
  data or DICOM overlay planes, cross-referenced against the study's own
  metadata to catch names a generic NER model wouldn't recognize.
- **`OrganMask`** — prostate-specific anatomical masking (whole gland,
  peripheral/transition zone, lesion ROI), for constraining downstream
  analysis to a segmented region.

Every transform appends a structured `AuditEntry` (step, action, detail,
confidence, `flagged_for_review`) instead of mutating data silently, so a
pipeline run produces one machine-readable report of exactly what changed.

```python
from prostate_deid import Compose, DicomTagScrub, Pseudonymize, OrganMask, Sample
from prostate_deid.organ_mask import WHOLE_GLAND

pipeline = Compose([
    DicomTagScrub(profile="hipaa_safe_harbor", always_apply=True),
    Pseudonymize(mapping_store="/secure/institution-only/mapping.json", always_apply=True),
    OrganMask(keep_labels=WHOLE_GLAND, always_apply=True),
])

result = pipeline(sample)  # sample: prostate_deid.Sample

for entry in result.audit:
    print(entry)
```

## Installation

```bash
pip install prostate-deid          # core: DICOM scrubbing, pseudonymization, masking
pip install prostate-deid[ocr]     # + burned-in text redaction (pytesseract, spaCy)
pip install prostate-deid[dashboard]  # + reviewer QA dashboard (Streamlit)
```

## Anonymization vs. pseudonymization

**This library performs pseudonymization, not anonymization.** `Pseudonymize`
deliberately retains a reversible mapping (patient ID → pseudo-ID, and a
per-patient date-shift offset) so that scans from the same patient at
different time points remain linkable — this is often a real research
requirement (e.g. PSA follow-up, treatment-response imaging), not an
oversight.

Under GDPR, retaining *any* mapping that could re-identify a patient — even
one held only by the originating institution, even encrypted — makes the
output pseudonymized data, not anonymized data, with its own legal
obligations (Art. 4(5)). Concretely:

- The `mapping_store` file must be written to a location **outside** the
  directory tree containing the de-identified images, and outside any
  repository or archive the de-identified data is shared through.
- Set `PROSTATE_DEID_KEY` (or pass `secret=`) to a real institution-managed
  secret — the default is intentionally empty and unsuitable for production.
- If your use case requires true anonymization (no retained mapping,
  independent per-study ID randomization, no date linkage), do not use
  `Pseudonymize` — irreversibly discard the identifying fields instead.

## Private DICOM tags

Standard confidentiality profiles (HIPAA Safe Harbor, DICOM PS3.15) only
cover the public DICOM tag dictionary. Scanner manufacturers routinely embed
PHI in undocumented private tags. **`DicomTagScrub` strips all unrecognized
private tags by default** (`strip_private_tags=True`) rather than attempting
to parse unknown vendor content — guessing wrong about an unknown private
tag is worse than dropping it.

## Human review workflow

`flagged_for_review=True` on an `AuditEntry` is not a dead end — see the
[`dashboard/`](dashboard/) reviewer QA tool, which loads a pipeline's audit
trail and surfaces every flagged region as a cropped thumbnail for quick
visual triage. An audit trail nobody reviews creates false confidence, which
is worse than not flagging at all.

## Compliance disclaimer

**Using this library does not, by itself, constitute HIPAA or GDPR
compliance.** It is a tool toward compliance, not a certification of it.
Actual regulatory compliance requires your institution's own review — e.g.
HIPAA's Expert Determination method, or a GDPR Data Protection Impact
Assessment (DPIA). This library is provided under the MIT License, which
disclaims all warranty; that disclaimer explicitly extends to any implied
claim of regulatory compliance.

## Validation status

This library has not yet been benchmarked against the [MIDI-B Challenge](https://arxiv.org/abs/2303.10473)
public dataset. Test fixtures currently use only synthetic data (fake names,
fake dates, synthetic prostate-shaped volumes) — see [`tests/`](tests/).
Real prostate MRI validation (e.g. against PROSTATEx, under its own TCIA data
use agreement) is tracked as future work, not yet performed. Do not treat the
current test suite passing as evidence of real-world detection recall.

## License

MIT — see [LICENSE](LICENSE).
