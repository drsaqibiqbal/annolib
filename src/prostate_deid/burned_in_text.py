"""Burned-in pixel PHI detection and redaction.

Detects text baked into pixel data (OCR), classifies PHI vs. non-PHI (NER),
and cross-references DICOM metadata (patient name, etc.) to catch entities a
generic NER model wouldn't recognize on its own -- e.g. a name that isn't a
common name in the NER model's training data, but does appear verbatim in
this study's own PatientName field.

Also scrubs DICOM overlay planes (0x60xx,3000), a separate bitmap layer from
the main pixel data that is easy to forget precisely because it "is not the
image" -- PHI has been found there in the wild (ultrasound/annotation
overlays in particular).

False negatives (missed PHI) and false positives (over-redaction) are not
symmetric: a missed detection is a privacy incident, an over-redaction is a
data-quality loss. Default thresholds bias toward flagging/redacting when
uncertain, per that stated policy -- not a shrug-worthy parameter default.

OCR/NER backends are optional dependencies (`pip install prostate-deid[ocr]`)
so the core package stays lightweight for users who only need DICOM tag
scrubbing or masking.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from .core import AuditEntry, Sample, Transform

BBox = Tuple[int, int, int, int]  # y0, y1, x0, x1


class BurnedInTextRedact(Transform):
    """Detects and blacks out burned-in PHI text in pixel data and overlays.

    ocr_engine: "tesseract" (via pytesseract) is the only backend wired up
        currently; pass a callable(image: np.ndarray) -> List[(bbox, text)]
        to use a custom engine instead.
    confidence_threshold: detections below this NER confidence are still
        redacted, but flagged_for_review=True -- biasing toward redaction
        under uncertainty rather than silently letting a low-confidence hit
        through unredacted.
    scrub_overlays: also scan and blackout DICOM overlay planes
        (sample.dicom_meta["_overlays"], a list of boolean/uint8 arrays), not
        just the main pixel array.
    """

    def __init__(
        self,
        ocr_engine: str = "tesseract",
        confidence_threshold: float = 0.6,
        scrub_overlays: bool = True,
        cross_reference_metadata: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.ocr_engine = ocr_engine
        self.confidence_threshold = confidence_threshold
        self.scrub_overlays = scrub_overlays
        self.cross_reference_metadata = cross_reference_metadata

    def apply(self, sample: Sample) -> Sample:
        detections = self._detect_phi_regions(sample.image, sample)
        for bbox, text, conf in detections:
            self._black_out(sample.image, bbox)
            sample.log(
                AuditEntry(
                    step="burned_in_phi",
                    action="redacted_region",
                    detail=f"bbox={bbox}, text_hash={hash(text)}",
                    confidence=conf,
                    flagged_for_review=conf < self.confidence_threshold,
                )
            )

        if self.scrub_overlays and sample.dicom_meta:
            overlays = sample.dicom_meta.get("_overlays")
            if overlays:
                for idx, overlay in enumerate(overlays):
                    overlay_detections = self._detect_phi_regions(overlay, sample)
                    for bbox, text, conf in overlay_detections:
                        self._black_out(overlay, bbox)
                        sample.log(
                            AuditEntry(
                                step="burned_in_phi",
                                action="redacted_overlay_region",
                                detail=f"overlay={idx}, bbox={bbox}, text_hash={hash(text)}",
                                confidence=conf,
                                flagged_for_review=conf < self.confidence_threshold,
                            )
                        )

        return sample

    def _detect_phi_regions(
        self, image: np.ndarray, sample: Sample
    ) -> List[Tuple[BBox, str, float]]:
        raw_detections = self._run_ocr(image)
        known_names = self._known_identifiers(sample)

        results: List[Tuple[BBox, str, float]] = []
        for bbox, text in raw_detections:
            conf = self._classify_phi(text, known_names)
            if conf is not None:
                results.append((bbox, text, conf))
        return results

    def _run_ocr(self, image: np.ndarray) -> List[Tuple[BBox, str]]:
        if self.ocr_engine != "tesseract":
            raise ValueError(f"Unsupported ocr_engine: {self.ocr_engine!r}")

        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "BurnedInTextRedact requires the 'ocr' extra: "
                "pip install prostate-deid[ocr]"
            ) from exc

        frame = image if image.ndim == 2 else image[tuple([0] * (image.ndim - 2))]
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(frame_uint8)

        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        detections: List[Tuple[BBox, str]] = []
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            detections.append(((y, y + h, x, x + w), text.strip()))
        return detections

    def _known_identifiers(self, sample: Sample) -> List[str]:
        if not self.cross_reference_metadata or not sample.dicom_meta:
            return []
        candidate_fields = [
            "PatientName",
            "PatientID",
            "ReferringPhysicianName",
            "InstitutionName",
        ]
        values = []
        for field_name in candidate_fields:
            value = sample.dicom_meta.get(field_name)
            if value:
                values.append(str(value))
        return values

    def _classify_phi(self, text: str, known_identifiers: List[str]) -> Optional[float]:
        for identifier in known_identifiers:
            normalized = identifier.replace("^", " ").strip().lower()
            if normalized and normalized in text.lower():
                return 0.99

        try:
            nlp = _get_spacy_pipeline()
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "DATE"}:
                    return 0.75
        except ImportError:
            pass

        return None

    @staticmethod
    def _black_out(image: np.ndarray, bbox: BBox) -> None:
        y0, y1, x0, x1 = bbox
        image[..., y0:y1, x0:x1] = 0


_SPACY_PIPELINE = None


def _get_spacy_pipeline():
    global _SPACY_PIPELINE
    if _SPACY_PIPELINE is None:
        import spacy

        try:
            _SPACY_PIPELINE = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise ImportError(
                "spaCy model 'en_core_web_sm' not installed. Run: "
                "python -m spacy download en_core_web_sm"
            ) from exc
    return _SPACY_PIPELINE
