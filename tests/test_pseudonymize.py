import numpy as np

from prostate_deid import Pseudonymize, Sample


def test_same_patient_id_maps_consistently(tmp_path, synthetic_dicom_meta):
    store_path = tmp_path / "mapping.json"

    sample_a = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta=dict(synthetic_dicom_meta))
    sample_b = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta=dict(synthetic_dicom_meta))

    transform_a = Pseudonymize(mapping_store=store_path, always_apply=True)
    result_a = transform_a(sample_a)

    transform_b = Pseudonymize(mapping_store=store_path, always_apply=True)
    result_b = transform_b(sample_b)

    assert result_a.dicom_meta["PatientID"] == result_b.dicom_meta["PatientID"]
    assert result_a.dicom_meta["PatientID"] != "SYNTH-0001"


def test_date_shift_is_consistent_across_fields(tmp_path, synthetic_dicom_meta):
    store_path = tmp_path / "mapping.json"
    sample = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta=dict(synthetic_dicom_meta))

    transform = Pseudonymize(mapping_store=store_path, always_apply=True)
    result = transform(sample)

    assert result.dicom_meta["StudyDate"] == result.dicom_meta["SeriesDate"]
    assert result.dicom_meta["StudyDate"] != "20260101"


def test_different_patients_get_different_pseudo_ids(tmp_path):
    store_path = tmp_path / "mapping.json"

    sample_a = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta={"PatientID": "REAL-A"})
    sample_b = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta={"PatientID": "REAL-B"})

    transform = Pseudonymize(mapping_store=store_path, always_apply=True)
    result_a = transform(sample_a)
    result_b = transform(sample_b)

    assert result_a.dicom_meta["PatientID"] != result_b.dicom_meta["PatientID"]


def test_mapping_persists_to_disk(tmp_path):
    store_path = tmp_path / "mapping.json"
    sample = Sample(image=np.zeros((1, 4, 4, 4)), dicom_meta={"PatientID": "REAL-A"})

    transform = Pseudonymize(mapping_store=store_path, always_apply=True)
    transform(sample)

    assert store_path.exists()


def test_unparseable_date_is_flagged_not_silently_dropped(tmp_path):
    store_path = tmp_path / "mapping.json"
    sample = Sample(
        image=np.zeros((1, 4, 4, 4)),
        dicom_meta={"PatientID": "REAL-A", "StudyDate": "not-a-date"},
    )

    transform = Pseudonymize(mapping_store=store_path, always_apply=True)
    result = transform(sample)

    flagged = result.flagged_entries()
    assert any(e.action == "date_shift_failed" for e in flagged)
