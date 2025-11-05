"""Test color banner and STAT output."""

from contextlib import contextmanager
from io import StringIO
import sys
import argparse
from typing import Any

import pytest

from dicomaster import show_banner, pretty_print_stat


@contextmanager
def captured_output():
    """Context manager to capture stdout for testing."""
    new_out = StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield sys.stdout
    finally:
        sys.stdout = old_out


def test_show_banner_quiet():
    """Test that banner is suppressed with quiet flag."""
    args = argparse.Namespace(quiet=True)
    with captured_output() as out:
        show_banner(args)
    assert out.getvalue() == ''


def test_show_banner_no_banner():
    """Test that banner is suppressed with no_banner flag."""
    args = argparse.Namespace(quiet=False, no_banner=True)
    with captured_output() as out:
        show_banner(args)
    assert out.getvalue() == ''


    def test_show_banner_fallback():
        """Test that banner falls back gracefully."""
        args = argparse.Namespace(quiet=False, no_banner=False)
        with captured_output() as out:
            show_banner(args)
        output = out.getvalue()
        assert 'DICOM' in output
        assert 'Santo Paul' in output
@pytest.mark.parametrize('color', [True, False])
def test_pretty_print_stat(color: bool):
    """Test pretty stat printing with and without color."""
    stat: dict[str, Any] = {
        'patient_age': '45Y',
        'patient_sex': 'M',
        'modality': 'MR',
        'body_part_examined': 'BRAIN',
        'study_date_time': '2023-01-01 15:30:00',
        'urgent': False,
        'urgent_reasons': []
    }
    full: dict[str, Any] = {}
    
    with captured_output() as out:
        pretty_print_stat(stat, full=full, color=color)
    
    output = out.getvalue().strip()
    assert '45Y' in output
    assert 'M' in output
    assert 'MR' in output
    assert 'BRAIN' in output
    assert '[OK]' in output


def test_pretty_print_stat_urgent():
    """Test pretty stat printing with urgent status."""
    stat: dict[str, Any] = {
        'patient_age': '45Y',
        'patient_sex': 'M',
        'modality': 'MR',
        'body_part_examined': 'BRAIN',
        'study_date_time': '2023-01-01 15:30:00',
        'urgent': True,
        'urgent_reasons': ['Contrast reaction', 'Critical finding']
    }
    full: dict[str, Any] = {}
    
    with captured_output() as out:
        pretty_print_stat(stat, full=full)
    
    output = out.getvalue().strip()
    assert '[URGENT]' in output
    assert 'Contrast reaction' in output
    assert 'Critical finding' in output


def test_pretty_print_stat_handles_missing():
    """Test pretty stat printing gracefully handles missing fields."""
    stat: dict[str, Any] = {}
    full: dict[str, Any] = {}
    
    with captured_output() as out:
        pretty_print_stat(stat, full=full)
    
    output = out.getvalue().strip()
    assert 'N/A' in output
    assert '[OK]' in output  # Default non-urgent