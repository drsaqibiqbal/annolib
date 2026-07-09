"""Consistent patient ID remapping and date shifting.

This module implements PSEUDONYMIZATION, not anonymization. A reversible
mapping is deliberately retained -- per-patient ID substitution and a
per-patient date-shift offset are both consistent across repeated calls for
the same source ID, so that longitudinal research (e.g. PSA follow-up,
repeat MRI across a treatment course) remains possible: scan A and scan B
for the same patient still link to each other after processing.

Under GDPR, retaining any mapping that could re-identify a patient -- even
one held only by the originating institution, even encrypted -- makes this
pseudonymization, a legally distinct category from anonymization with its
own obligations (Art. 4(5), recital 26). This library does not claim to
produce anonymized data, and the mapping store this class writes to must
never be shipped alongside, or stored in the same repository/location as,
the de-identified output.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from .core import AuditEntry, Sample, Transform

_DATE_FIELDS_DEFAULT = ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"]
_ID_FIELDS_DEFAULT = ["PatientID"]
# UIDs are themselves identifiers (they encode institution/device roots and
# are globally unique per study), so they must be regenerated -- but
# consistently, so intra-study cross-references (a series pointing at its
# study) survive. Same source UID -> same new UID, deterministically.
_UID_FIELDS_DEFAULT = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]

# Bound consistent date shift to a wide but finite range so shifted dates
# can't accidentally collide with a plausible real date range far outside
# the source data's era.
_MAX_SHIFT_DAYS = 365 * 20


class MappingStore:
    """Reads/writes the patient_id -> (pseudo_id, shift_days) mapping table.

    Intentionally simple (a local JSON file) rather than a database, so the
    file's location is an explicit, visible decision the caller makes -- and
    can point it somewhere outside the de-identified output tree, e.g. an
    encrypted institution-side volume.
    """

    def __init__(self, path: str | Path, secret: Optional[str] = None):
        self.path = Path(path)
        self.secret = secret or os.environ.get("PROSTATE_DEID_KEY", "")
        self._table: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            self._table = json.loads(self.path.read_text())

    def _derive_pseudo_id(self, patient_id: str) -> str:
        digest = hmac.new(
            self.secret.encode("utf-8"),
            patient_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"PSEUDO-{digest[:16].upper()}"

    def _derive_shift_days(self, patient_id: str) -> int:
        digest = hmac.new(
            self.secret.encode("utf-8"),
            f"shift:{patient_id}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        offset = int(digest[:8], 16) % (2 * _MAX_SHIFT_DAYS)
        return offset - _MAX_SHIFT_DAYS

    def resolve(self, patient_id: str) -> Dict[str, Any]:
        if patient_id not in self._table:
            self._table[patient_id] = {
                "pseudo_id": self._derive_pseudo_id(patient_id),
                "shift_days": self._derive_shift_days(patient_id),
            }
        return self._table[patient_id]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._table, indent=2, sort_keys=True))


def _shift_date_str(value: str, shift_days: int, fmt: str = "%Y%m%d") -> str:
    parsed = datetime.strptime(value, fmt).date()
    shifted = parsed + timedelta(days=shift_days)
    return shifted.strftime(fmt)


def _derive_uid(source_uid: str, secret: str, root: str = "2.25") -> str:
    """Deterministically derive a replacement DICOM UID from a source UID.

    Uses the 2.25 root (a registered arc for UUID-derived UIDs) followed by
    an integer derived from HMAC(secret, source_uid), so the same source UID
    always maps to the same replacement -- preserving intra-study references
    -- without leaking the original institution/device UID roots.
    """
    digest = hmac.new(secret.encode("utf-8"), source_uid.encode("utf-8"), hashlib.sha256).hexdigest()
    # Take 32 hex chars (128 bits) as an integer, matching UUID-derived width.
    numeric = int(digest[:32], 16)
    uid = f"{root}.{numeric}"
    return uid[:64]  # DICOM UIDs are capped at 64 characters


class Pseudonymize(Transform):
    """Remaps patient IDs and shifts date fields consistently per patient.

    mapping_store: a MappingStore instance (or a path, which will be wrapped
        in one). Must point outside the output directory for de-identified
        images -- this class does not enforce that, it is a deployment
        responsibility documented here and in the README.
    id_fields / date_fields: dicom_meta keys to remap / shift. Unrecognized
        or missing fields are skipped silently (not every study has every
        field populated).
    """

    def __init__(
        self,
        mapping_store: "MappingStore | str | Path",
        id_fields: Optional[list] = None,
        date_fields: Optional[list] = None,
        uid_fields: Optional[list] = None,
        **kwargs: Any,
    ):
        Transform.__init__(self, **kwargs)
        if isinstance(mapping_store, (str, Path)):
            mapping_store = MappingStore(mapping_store)
        self.mapping_store = mapping_store
        self.id_fields = id_fields or list(_ID_FIELDS_DEFAULT)
        self.date_fields = date_fields or list(_DATE_FIELDS_DEFAULT)
        self.uid_fields = uid_fields if uid_fields is not None else list(_UID_FIELDS_DEFAULT)

    def apply(self, sample: Sample) -> Sample:
        if sample.dicom_meta is None:
            return sample

        patient_id = None
        for field_name in self.id_fields:
            if field_name in sample.dicom_meta:
                patient_id = sample.dicom_meta[field_name]
                break

        if patient_id is None:
            return sample

        mapping = self.mapping_store.resolve(str(patient_id))

        for field_name in self.id_fields:
            if field_name in sample.dicom_meta:
                sample.dicom_meta[field_name] = mapping["pseudo_id"]
                sample.log(
                    AuditEntry(
                        step="pseudonymize",
                        action="remapped_id",
                        detail=field_name,
                    )
                )

        for field_name in self.date_fields:
            if field_name in sample.dicom_meta:
                try:
                    sample.dicom_meta[field_name] = _shift_date_str(
                        sample.dicom_meta[field_name], mapping["shift_days"]
                    )
                    sample.log(
                        AuditEntry(
                            step="pseudonymize",
                            action="shifted_date",
                            detail=field_name,
                        )
                    )
                except ValueError:
                    sample.log(
                        AuditEntry(
                            step="pseudonymize",
                            action="date_shift_failed",
                            detail=f"{field_name}: unparseable value",
                            flagged_for_review=True,
                        )
                    )

        for field_name in self.uid_fields:
            if field_name in sample.dicom_meta:
                sample.dicom_meta[field_name] = _derive_uid(
                    sample.dicom_meta[field_name], self.mapping_store.secret
                )
                sample.log(
                    AuditEntry(
                        step="pseudonymize",
                        action="regenerated_uid",
                        detail=field_name,
                    )
                )

        self.mapping_store.save()
        return sample
