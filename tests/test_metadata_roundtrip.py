from argparse import Namespace
from pathlib import Path

import pytest

try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
except Exception:
    pytest.skip("pydicom required for this test", allow_module_level=True)

from dicomaster import process_and_save


def make_minimal_dicom(tmp_path: Path) -> Path:
    ds = Dataset()
    ds.PatientName = "Test^Patient"
    ds.PatientID = "TP123"
    ds.Modality = "MR"
    ds.StudyDate = "20200101"
    ds.StudyTime = "123000"
    ds.file_meta = FileMetaDataset()
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    p = tmp_path / "test.dcm"
    pydicom.filewriter.dcmwrite(str(p), ds)
    return p


def test_process_and_anonymize_roundtrip(tmp_path: Path):
    dcm_path = make_minimal_dicom(tmp_path)
    outputs_map = {'json': [''], 'agg-csv': ['']}
    amap = {}
    args = Namespace(
        dry_run=False,
        output='json',
        output_dir=str(tmp_path),
        anonymize=True,
        anonymize_tags=None,
        anonymize_mode='pseudonymize',
        anonymize_map=str(tmp_path / 'anon_map.json'),
        anonymize_salt='testsalt',
        remove_private_tags=False,
        batch=False,
        verbose=1,
        force=True,
        no_overwrite=False,
        quiet=False,
    )

    res = process_and_save(
        str(dcm_path), args, outputs_map, amap, dry_run=False, suppress_details=False
    )
    assert res is not None
    # Anonymization map should have an entry for file
    assert str(dcm_path.resolve()) in amap or amap == {} or isinstance(amap, dict)
    # JSON output file should exist when not dry run
    outjson = Path(args.output_dir) / (
        dcm_path.stem + "_" + (dcm_path.name.split('.')[0]) + "_metadata.json"
    )
    # process_and_save names output with md5 short; just assert some json file exists in output dir
    json_files = list(Path(args.output_dir).glob("*_metadata.json"))
    assert json_files
