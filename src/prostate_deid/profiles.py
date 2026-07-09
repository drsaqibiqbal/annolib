"""Named DICOM tag-removal profiles.

Tag lists are keyword-based (pydicom keyword form) rather than raw (group,
element) tuples, since keywords are what a reviewer can sanity-check without
a data dictionary open.

hipaa_safe_harbor mirrors the 18 HIPAA identifier categories as they map onto
standard DICOM header fields. dicom_ps3.15_basic is a minimal subset of the
DICOM PS3.15 Basic Application Level Confidentiality Profile.

These lists are deliberately conservative starting points, not a certified
mapping to either standard -- see the README's compliance disclaimer.
"""

from __future__ import annotations

from typing import Dict, List

HIPAA_SAFE_HARBOR: List[str] = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "OtherPatientIDsSequence",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName",
    "PhysiciansOfRecord",
    "OperatorsName",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "StationName",
    "StudyID",
    "AccessionNumber",
    "AdmittingDiagnosesDescription",
    "PatientAge",
    "EthnicGroup",
    "PersonName",
    "RequestingPhysician",
    "ScheduledPerformingPhysicianName",
    "DeviceSerialNumber",
]

DICOM_PS3_15_BASIC: List[str] = [
    "PatientName",
    "PatientID",
    "OtherPatientIDs",
    "OtherPatientNames",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientAddress",
]

PROFILES: Dict[str, List[str]] = {
    "hipaa_safe_harbor": HIPAA_SAFE_HARBOR,
    "dicom_ps3.15_basic": DICOM_PS3_15_BASIC,
}


def resolve_profile(name: str) -> List[str]:
    if name not in PROFILES:
        raise ValueError(
            f"Unknown profile '{name}'. Available: {sorted(PROFILES)}, "
            "or pass a custom list via extra_tags_to_remove."
        )
    return list(PROFILES[name])
