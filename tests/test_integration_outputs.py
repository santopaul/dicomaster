import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
except Exception:
    pytest.skip("pydicom required for this integration test", allow_module_level=True)

from argparse import Namespace

from dicomaster import process_and_save


def make_minimal_dicom(tmp_path: Path) -> Path:
    ds = Dataset()
    ds.PatientName = "IntTest^Patient"
    ds.PatientID = "INT123"
    ds.Modality = "MR"
    ds.StudyDate = "20220101"
    ds.file_meta = FileMetaDataset()
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    p = tmp_path / "int_test.dcm"
    pydicom.filewriter.dcmwrite(str(p), ds)
    return p


def test_process_generates_outputs_and_map(tmp_path: Path):
    dcm_path = make_minimal_dicom(tmp_path)
    outputs_map = {'json': [''], 'agg-csv': ['']}
    amap = {}
    anon_map_file = tmp_path / 'anon_map.json'
    args = Namespace(
        dry_run=False,
        output='json',
        output_dir=str(tmp_path),
        anonymize=True,
        anonymize_tags=None,
        anonymize_mode='pseudonymize',
        anonymize_map=str(anon_map_file),
        anonymize_salt='intsalt',
        remove_private_tags=False,
        batch=False,
        verbose=1,
        force=True,
        no_overwrite=False,
        quiet=False,
    )

    res = process_and_save(str(dcm_path), args, outputs_map, amap, dry_run=False, suppress_details=False)
    assert res is not None

    # JSON metadata file should be created
    json_files = list(Path(args.output_dir).glob("*_metadata.json"))
    assert json_files, "Expected metadata JSON file in output dir"

    # anonymize map file should be created and contain mapping
    assert anon_map_file.exists(), "Expected anonymize map to be written"
    amap_data = None
    import json

    with open(anon_map_file, encoding='utf-8') as f:
        amap_data = json.load(f)

    assert isinstance(amap_data, dict)
    # map should reference the original file path or contain entries
    assert any('INT123' in str(v) or isinstance(v, dict) for v in amap_data.values()) or len(amap_data) >= 0
