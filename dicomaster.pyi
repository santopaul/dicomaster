"""Type stubs for dicomaster.py"""
import argparse
from typing import Any

# Note: These are structural types for type checking only
PyDicomDataset = Any  # Represents pydicom.dataset.Dataset
ColoredText = str     # Represents termcolor.colored output

def safe_colored(text: str, color: str | None = None, on_color: str | None = None,
                attrs: list[str] | None = None) -> str: ...

def show_banner(args: argparse.Namespace | bool = True, font: str | None = None) -> None: ...

def process_and_save(path: str, args: argparse.Namespace,
                    mappings: dict[str, Any] | None = None) -> dict[str, Any] | None: ...

def try_thumbnail(ds: PyDicomDataset, out_path: str, max_size: int = 256) -> bool: ...

def save_pixel_images(ds: PyDicomDataset, out_prefix: str, ext: str = '.png') -> list[str]: ...

def get_private_tags(ds: PyDicomDataset) -> list[dict[str, Any]]: ...

def pretty_print_stat(stat: dict[str, Any], full: dict[str, Any],
                     color: bool = True, detail: bool = False) -> None: ...

def create_html_report(stat: dict[str, Any], full: dict[str, Any],
                      thumbnail_path: str | None = None) -> str: ...

def create_text_report(stat: dict[str, Any], full: dict[str, Any],
                      urgent: bool = False, urgent_reasons: list[str] = [],
                      phi_flags: list[str] = [], private: list[dict[str, Any]] = []) -> str: ...
