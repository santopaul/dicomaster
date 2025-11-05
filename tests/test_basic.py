import sys
from pathlib import Path

# Ensure project root is on sys.path so tests can import the module when run from pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dicomaster import apply_anonymization_to_sanitized, pseudonymize_value


def test_pseudonymize_value_returns_token_and_salt():
    token, salt = pseudonymize_value('John Doe', salt=None)
    assert token is not None and token.startswith('anon_')
    # salt should be a hex string when generated
    assert isinstance(salt, str)


def test_apply_anonymization_updates_map():
    sanitized = {'patient_name': 'Jane Doe', 'patient_id': '12345'}
    run_map = {}
    sanitized_after, per_file_map = apply_anonymization_to_sanitized(dict(sanitized), ['patient_name', 'patient_id'], 'pseudonymize', 'testsalt', run_map, '/tmp/fake.dcm')
    # sanitized should be modified
    assert sanitized_after.get('patient_name', '').startswith('anon_')
    assert '/tmp/fake.dcm' in run_map
    entry = run_map['/tmp/fake.dcm']
    assert 'patient_name' in entry
    # pseudonymize mode should produce a pseudonym entry
    assert 'pseudonym' in entry['patient_name']
