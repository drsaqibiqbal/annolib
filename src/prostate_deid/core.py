"""Core primitives: Sample, AuditEntry, Transform, Compose.

Mirrors the Transform/Compose contract established by volumetric augmentation
libraries (e.g. Bio-Volumentations), so users already familiar with that
pattern can adopt this library directly. The key difference: every Transform
here appends to a shared audit trail instead of mutating data silently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AuditEntry:
    """One record of a single change made to a Sample by a Transform."""

    step: str
    action: str
    detail: str
    confidence: Optional[float] = None
    flagged_for_review: bool = False


@dataclass
class Sample:
    """The unit that flows through a pipeline.

    image: volumetric array, convention CZYX(T).
    dicom_meta: parsed DICOM header fields (e.g. from pydicom.Dataset), as a
        plain dict so Transforms don't depend on pydicom internals directly.
    mask: anatomical mask (e.g. prostate gland / zone / lesion), same spatial
        convention as image, values distinguish region labels (0 = outside).
    """

    image: np.ndarray
    dicom_meta: Optional[Dict[str, Any]] = None
    mask: Optional[np.ndarray] = None
    audit: List[AuditEntry] = field(default_factory=list)

    def log(self, entry: AuditEntry) -> None:
        self.audit.append(entry)

    def flagged_entries(self) -> List[AuditEntry]:
        return [e for e in self.audit if e.flagged_for_review]


class Transform(ABC):
    """Base class for all de-identification / masking operations."""

    def __init__(self, p: float = 1.0, always_apply: bool = False):
        self.p = 1.0 if always_apply else p

    @abstractmethod
    def apply(self, sample: Sample) -> Sample:
        ...

    def __call__(self, sample: Sample) -> Sample:
        if np.random.rand() <= self.p:
            return self.apply(sample)
        return sample


class Compose:
    """Runs a list of Transforms in sequence, accumulating one shared audit trail."""

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample
