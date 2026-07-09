from prostate_deid import DicomTagScrub, Sample
import numpy as np


def test_removes_hipaa_safe_harbor_fields(synthetic_dicom_meta):
    sample = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta=dict(synthetic_dicom_meta))
    transform = DicomTagScrub(profile="hipaa_safe_harbor", always_apply=True)

    result = transform(sample)

    assert "PatientName" not in result.dicom_meta
    assert "PatientID" not in result.dicom_meta
    assert "ReferringPhysicianName" not in result.dicom_meta
    assert "InstitutionName" not in result.dicom_meta


def test_keeps_non_identifying_fields(synthetic_dicom_meta):
    sample = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta=dict(synthetic_dicom_meta))
    transform = DicomTagScrub(profile="hipaa_safe_harbor", always_apply=True)

    result = transform(sample)

    assert result.dicom_meta.get("Manufacturer") == "SyntheticVendor"


def test_strips_private_tags_by_default(synthetic_dicom_meta):
    sample = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta=dict(synthetic_dicom_meta))
    transform = DicomTagScrub(profile="hipaa_safe_harbor", always_apply=True)

    result = transform(sample)

    assert "0x00410010" not in result.dicom_meta


def test_logs_audit_entries_for_every_removed_tag(synthetic_dicom_meta):
    sample = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta=dict(synthetic_dicom_meta))
    transform = DicomTagScrub(profile="hipaa_safe_harbor", always_apply=True)

    result = transform(sample)

    removed_details = {e.detail for e in result.audit if e.action == "removed_tag"}
    assert "PatientName" in removed_details


def test_unknown_profile_raises():
    import pytest
    from prostate_deid import DicomTagScrub

    sample = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta={"PatientName": "x"})
    transform = DicomTagScrub(profile="not_a_real_profile", always_apply=True)

    with pytest.raises(ValueError):
        transform(sample)


def test_exclude_tags_keeps_pseudonymized_fields(synthetic_dicom_meta):
    sample = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta=dict(synthetic_dicom_meta))
    transform = DicomTagScrub(
        profile="hipaa_safe_harbor",
        exclude_tags=["PatientID", "StudyDate"],
        always_apply=True,
    )

    result = transform(sample)

    assert result.dicom_meta["PatientID"] == "SYNTH-0001"
    assert result.dicom_meta["StudyDate"] == "20260101"
    assert "PatientName" not in result.dicom_meta


def test_no_dicom_meta_is_noop():
    sample = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta=None)
    transform = DicomTagScrub(always_apply=True)

    result = transform(sample)

    assert result.dicom_meta is None
    assert result.audit == []
