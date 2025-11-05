#!/usr/bin/env python3
"""
dicomaster — Robust DICOM metadata extractor & anonymizer (release polish)

This file incorporates the prioritized fixes you requested:
 - PBKDF2 pseudonymization (cryptography optional) with correct use of KDF output
 - Non-verbose salt handling (generated salt saved in map, not fully logged)
 - Per-file anonymization mapping structure: {"/abs/path": {"tag": {"orig":..., "pseudonym":..., "salt":...}}}
 - Thread cap based on os.cpu_count()*2 and MAX_THREAD_CAP
 - No double file rescans; total_files computed once
 - FHIR mapping uses snake_case keys (study_instance_uid)
 - Schema field names aligned (study_date_time)
 - find_dicom_files includes .dcm/.DCM/.ima/.img
 - Streaming aggregation flushes periodically
 - Simpler progress logging frequency (every 10% or 10 files)
 - Salt length warning when cryptography available and salt short
 - Quiet mode truly suppresses info-level progress

Usage: see argparse help (``python dicom_tool.py -h``)
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import threading
import traceback
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    cast,
)

# Type variables for generic use
T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

# Define runtime types for imports
PILImage = Any
PILDraw = Any
PILFont = Any
NDArray = Any
DicomDataset = Any
NumPyModule = Any

# Runtime module references
np: NumPyModule | None = None
pillow: Any | None = None
pydicom: Any | None = None

if TYPE_CHECKING:
    # Protocol definitions for type checking
    class PILImage(Protocol):
        """PIL Image protocol"""

        def save(self, fp: str, format: str | None = None) -> None: ...
        def paste(self, im: PILImage, box: tuple[int, int]) -> None: ...
        def thumbnail(self, size: tuple[int, int]) -> None: ...

    class PILDraw(Protocol):
        """PIL ImageDraw protocol"""

        def text(
            self,
            xy: tuple[int, int],
            text: str,
            fill: str | None = None,
            font: PILFont | None = None,
        ) -> None: ...

    class PILFont(Protocol):
        """PIL ImageFont protocol"""

        def getsize(self, text: str) -> tuple[int, int]: ...

    class NDArray(Protocol):
        """NumPy array protocol"""

        @property
        def ndim(self) -> int: ...
        @property
        def shape(self) -> tuple[int, ...]: ...
        @property
        def dtype(self) -> Any: ...
        def __getitem__(self, key: int | slice) -> NDArray: ...
        def __len__(self) -> int: ...
        def astype(self, dtype: Any) -> NDArray: ...
        def __mul__(self, other: int | float) -> NDArray: ...
        def min(self) -> int | float: ...
        def max(self) -> int | float: ...
        def __sub__(self, other: int | float) -> NDArray: ...
        def __truediv__(self, other: int | float) -> NDArray: ...

    class NumPyModule(Protocol):
        """NumPy module protocol"""

        @property
        def uint8(self) -> Any: ...
        def asarray(self, array: Any) -> NDArray: ...

    class DicomDataset(Protocol):
        """DICOM dataset protocol"""

        @property
        def pixel_array(self) -> NDArray: ...
        def __getitem__(self, key: str) -> Any: ...
        def __setitem__(self, key: str, value: Any) -> None: ...


class DicomModule(Protocol):
    """Protocol for pydicom module interface"""

    def dcmread(self, fp: str, force: bool = ...) -> DicomDataset: ...
    def dcmwrite(self, filename: str, dataset: DicomDataset) -> None: ...


class ImageModule(Protocol):
    """Protocol for PIL.Image module interface"""

    def open(self, fp: str, mode: str = ...) -> PILImage: ...
    def new(self, mode: str, size: tuple[int, int], color: str = ...) -> PILImage: ...


class ImageDrawModule(Protocol):
    """Protocol for PIL.ImageDraw module interface"""

    def Draw(self, image: PILImage) -> PILDraw: ...


class ImageFontModule(Protocol):
    """Protocol for PIL.ImageFont module interface"""

    def truetype(self, font: str, size: int) -> PILFont: ...


# Initialize module-level variables with protocol types
np: NumPyModule | None = None
pil_image: ImageModule | None = None
pil_draw: ImageDrawModule | None = None
pil_font: ImageFontModule | None = None
pydicom_module: DicomModule | None = None

# Initialize availability flags
has_pil = False
has_numpy = False
has_pydicom = False
has_pyfiglet = False
has_cryptography = False

# Import optional dependencies
try:
    from PIL import Image, ImageDraw, ImageFont

    pil_image = cast(ImageModule, Image)
    pil_draw = cast(ImageDrawModule, ImageDraw)
    pil_font = cast(ImageFontModule, ImageFont)
    has_pil = True
except ImportError:
    pass

try:
    import numpy as np

    np = cast(NumPyModule, np)
    has_numpy = True
except ImportError:
    pass
except ImportError:
    _NP_AVAILABLE = False
    NDArray = Any  # For type checking

# Type aliases for clarity
StrDict = dict[str, str]
AnyDict = dict[str, Any]

try:
    import pydicom
    from pydicom.dataset import Dataset as PyDicomDataset  # For type hints
    from pydicom.errors import InvalidDicomError

    _PYDICOM_AVAILABLE = True
except ImportError:
    pydicom = None  # type: ignore
    PyDicomDataset = object  # type: ignore
    InvalidDicomError = Exception
    _PYDICOM_AVAILABLE = False

# Banner/color dependencies
try:
    import pyfiglet
    from termcolor import COLORS as TERM_COLORS
    from termcolor import colored

    _BANNER_AVAILABLE = True

    def safe_colored(
        text: str,
        color: str | None = None,
        on_color: str | None = None,
        attrs: list[str] | None = None,
    ) -> str:
        """Type-safe wrapper for termcolor.colored that handles missing deps gracefully.

        Args:
            text: The text to color
            color: Color name (red, green, yellow, blue, magenta, cyan, white)
            on_color: Background color (on_red, on_green, etc)
            attrs: Text attributes [bold, dark, underline, blink, reverse, concealed]

        Returns:
            Colored string if termcolor available, original string otherwise
        """
        if not color or color not in TERM_COLORS:
            return text
        try:
            result = colored(text, color, on_color, attrs)
            return cast(str, result)
        except Exception:
            return text

except ImportError:
    _BANNER_AVAILABLE = False
    TERM_COLORS = frozenset()  # type: ignore

    def safe_colored(
        text: str,
        color: str | None = None,
        on_color: str | None = None,
        attrs: list[str] | None = None,
    ) -> str:
        """Fallback when termcolor not available."""
        return text


# Optional heavy deps
try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

try:
    from tqdm import tqdm

    _TQDM = True
except Exception:
    _TQDM = False

try:
    from dateutil.relativedelta import relativedelta

    _RELATIVEDELTA_AVAILABLE = True
except Exception:
    _RELATIVEDELTA_AVAILABLE = False

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except Exception:
    _PANDAS_AVAILABLE = False

# cryptography optional for PBKDF2
_CRYPTO_AVAILABLE = False
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    _CRYPTO_AVAILABLE = True
except Exception:
    _CRYPTO_AVAILABLE = False

# banner optional
# Banner libs optional
try:
    import pyfiglet
    from termcolor import colored

    _BANNER_AVAILABLE = True
except ImportError:
    _BANNER_AVAILABLE = False

# colorama optional for colored logs (non-critical)
try:
    from colorama import Fore, Style
    from colorama import init as colorama_init

    colorama_init(autoreset=True)
except Exception:

    class _Dummy:
        def __getattr__(self, _):
            return ''

    Fore = Style = _Dummy()

# Constants
__version__ = "0.9.0"
DEFAULT_THREADS = 4
MAX_THREAD_CAP = 64
CRITICAL_KEYS = {
    "patient_id",
    "patient_name",
    "patient_age",
    "patient_sex",
    "modality",
    "body_part_examined",
    "study_description",
    "study_date_time",
}
DEFAULT_ANON_TAGS = [
    'patient_name',
    'patient_id',
    'patient_birth_date',
    'patient_birth_time',
    'patient_age',
    'patient_address',
    'other_patient_ids',
    'other_patient_ids_sequence',
    'other_patient_names',
    'referring_physician_name',
    'performing_physician_name',
    'operators_name',
    'institution_name',
    'station_name',
    'accession_number',
    'study_id',
    'series_description',
    'study_comments',
]

SUPPORTED_TYPES = {
    'json',
    'csv',
    'html',
    'image',
    'thumbnail',
    'fhir',
    'report',
    'agg-csv',
    'agg-json',
}
EXT_TO_TYPE = {
    'json': 'json',
    'csv': 'csv',
    'html': 'html',
    'png': 'image',
    'jpg': 'image',
    'jpeg': 'image',
    'bmp': 'image',
    'tiff': 'image',
    'tif': 'image',
    'thumb': 'thumbnail',
    'fhir': 'fhir',
}

# Lock to protect shared anonymization mapping written by worker threads
ANON_MAP_LOCK = threading.Lock()

# ----------------------- Arg parsing with groups --------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dicomaster", description="DICOM metadata extractor & fast batch aggregator"
    )
    p.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to DICOM file or folder. Use '-' to read file paths from stdin. Omit to enter interactive REPL.",
    )

    # Output group
    g_out = p.add_argument_group('Output options')
    g_out.add_argument(
        "-o",
        "--output",
        action="append",
        default=[],
        help="Output targets. Examples: json,csv,image,report,thumbnail,fhir,agg-csv,agg-json or filenames (report.png). Can repeat.",
    )
    g_out.add_argument(
        "--output-dir", default="./dicom_reports", help="Directory to save outputs when requested"
    )
    g_out.add_argument(
        "--no-overwrite", action="store_true", help="Don't overwrite existing output files"
    )
    g_out.add_argument(
        "--dry-run", action="store_true", help="Do a trial run without writing files"
    )

    # Anonymization group
    g_anon = p.add_argument_group('Anonymization options')
    g_anon.add_argument(
        "--anonymize",
        action="store_true",
        help="Enable anonymization (default set if --anonymize-tags not provided)",
    )
    g_anon.add_argument(
        "--anonymize-tags",
        type=str,
        default=None,
        help="Comma-separated tags to anonymize only (e.g. PatientName,PatientID). If omitted, default set is used.",
    )
    g_anon.add_argument(
        "--anonymize-mode",
        choices=['pseudonymize', 'remove'],
        default='pseudonymize',
        help="Pseudonymize (hash) or remove (blank) selected tags. Default: pseudonymize",
    )
    g_anon.add_argument(
        "--anonymize-map",
        type=str,
        default=None,
        help="Path to save pseudonymization map (JSON) when pseudonymize mode used",
    )
    g_anon.add_argument(
        "--anonymize-salt",
        type=str,
        default=None,
        help="Optional salt for pseudonym hashing (recommended for reproducibility)",
    )
    g_anon.add_argument(
        "--remove-private-tags",
        action="store_true",
        help="Remove private tags from outputs (safe default when anonymizing)",
    )

    # Performance & batch group
    g_perf = p.add_argument_group('Batch & performance')
    g_perf.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="Treat path as directory and scan recursively for .dcm files",
    )
    g_perf.add_argument(
        "-t", "--threads", type=int, default=DEFAULT_THREADS, help="Threads for batch processing"
    )
    g_perf.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max recursion depth when scanning folders (None = unlimited)",
    )
    g_perf.add_argument(
        "--min-progress-report",
        type=int,
        default=50,
        help="If processing more than this many files, suppress per-file metadata prints and show progress only",
    )
    g_perf.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds for each worker future (optional)",
    )

    # Misc
    g_misc = p.add_argument_group('Misc')
    g_misc.add_argument(
        "--force",
        action="store_true",
        help="Force read even if file meta missing (use with caution)",
    )
    g_misc.add_argument(
        "--show-private-values",
        action="store_true",
        help="Show full private tag values (may contain PHI)",
    )
    g_misc.add_argument("--minimal", action="store_true", help="Only show STAT quick summary")
    group = g_misc.add_mutually_exclusive_group()
    group.add_argument("--detail", action="store_true", help="Show detailed technical information")
    group.add_argument(
        "--full", action="store_true", help="Show complete technical details including all metadata"
    )
    g_misc.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet mode (suppress non-critical prints)"
    )
    g_misc.add_argument("-v", "--verbose", action="count", default=0, help="Verbose mode (-v, -vv)")
    g_misc.add_argument("--log-file", type=str, default=None, help="Optional log file path")
    g_misc.add_argument(
        "--check-deps", action="store_true", help="Check for optional dependencies and exit"
    )
    g_misc.add_argument("--version", action="store_true", help="Show version and exit")
    g_misc.add_argument(
        "--no-interactive", action="store_true", help="Do not prompt/REPL (useful for automation)"
    )
    g_misc.add_argument(
        "--export-schema",
        nargs='?',
        const='dicom_schema.csv',
        help="Export a CSV header template for Kaggle-style aggregation (optional filename)",
    )
    g_misc.add_argument("--no-banner", action="store_true", help="Skip banner display")
    return p


# ------------------------------- Logging ----------------------------------


def configure_logging(quiet: bool, verbose_count: int, log_file: str | None = None):
    if quiet:
        level = logging.ERROR
    else:
        if verbose_count >= 2:
            level = logging.DEBUG
        elif verbose_count == 1:
            level = logging.INFO
        else:
            level = logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        fh = logging.FileHandler(log_file)
        handlers.append(fh)
    logging.basicConfig(
        level=level, format='%(asctime)s %(levelname)s: %(message)s', handlers=handlers
    )
    logging.debug(
        "Logging initialized. level=%s, log_file=%s", logging.getLevelName(level), log_file
    )


# ----------------------- banner -----------------------


def show_banner(args: argparse.Namespace) -> None:
    """Display a colorful program banner with optional ASCII art.

    Args:
        args: Command line arguments including quiet/no_banner flags
    """
    if getattr(args, 'quiet', False) or getattr(args, 'no_banner', False):
        return

    name = 'Santo Paul'
    try:
        # Best-effort: if banner libs missing, try to import on the fly (never fatal)
        from typing import Any as _Any
        local_colored: _Any = None
        local_pyfiglet: _Any = None
        pyf = globals().get('pyfiglet', None)
        if _BANNER_AVAILABLE and pyf is not None:
            local_colored = safe_colored  # reuse
            local_pyfiglet = pyf
        else:
            try:
                import importlib
                local_pyfiglet = importlib.import_module('pyfiglet')
                from termcolor import colored as _colored  # type: ignore

                def local_colored(t: str, color: str | None = None, attrs: list[str] | None = None) -> str:  # type: ignore
                    return _colored(t, color, attrs=attrs)
            except Exception:
                # Best-effort auto-install for banner deps; never fatal
                try:
                    import subprocess
                    import sys as _sys
                    subprocess.run(
                        [_sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-q', 'pyfiglet', 'termcolor', 'colorama'],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    import importlib
                    local_pyfiglet = importlib.import_module('pyfiglet')
                    from termcolor import colored as _colored  # type: ignore

                    def local_colored(t: str, color: str | None = None, attrs: list[str] | None = None) -> str:  # type: ignore
                        return _colored(t, color, attrs=attrs)
                except Exception:
                    local_pyfiglet = None
                    local_colored = None

        if local_pyfiglet is not None and local_colored is not None:
            # Create ASCII art banner with a straight (non-slanted) font
            art = local_pyfiglet.figlet_format('Dicomaster', font='standard')
            print(local_colored(art, 'green', attrs=['bold']))
            # Single-line tagline: everything bold except the word 'by'
            vpart = local_colored(f"v{__version__}", 'cyan', attrs=['bold'])
            mid = local_colored(" — Secure DICOM Anonymizer & Batch Processor ", 'white', attrs=['bold'])
            by = local_colored("by ", 'white')  # intentionally not bold
            npart = local_colored(name, 'yellow', attrs=['bold'])
            print(vpart + mid + by + npart)
            print()  # spacing
            return

    except Exception as e:
        logging.debug("Banner error (non-fatal): %s", e)

    # Simple fallback banner (no color libs available) single-line
    print(f"*** Dicomaster v{__version__} — Secure DICOM Anonymizer & Batch Processor by {name} ***")


def pretty_print_stat(
    stat: dict[str, Any],
    full: dict[str, Any] | None = None,
    color: bool = True,
    detail: bool = False,
) -> None:
    """Print a compact, colorful STAT summary to stdout with optional detailed view.

    Args:
        stat: Dictionary containing critical DICOM metadata fields
        full: Optional full metadata dictionary for detailed view
        color: Whether to use colored output (requires termcolor)
        detail: Whether to show detailed information after the summary

    Summary includes:
    - Patient age and sex (yellow)
    - Modality and body part (cyan)
    - Study date/time (green)
    - Study description and series info
    - Accession and study IDs
    - Urgent status (red/green) and reasons (red)

    Detailed view adds:
    - Station name and technical details
    - Image dimensions and bit depth
    - Scanning parameters (TR/TE)
    - Protocol information
    - PHI warnings if present
    """
    try:
        # Helper for safe coloring with fallback to plain text
        def cc(text: object, col: str | None = None, attrs: list[str] | None = None) -> str:
            t = str(text)
            return safe_colored(t, col, attrs=attrs) if color and col else t

        def b(text: object) -> str:
            return cc(text, 'white', ['bold']) if color else str(text)

        # Prepare critical fields with safe defaults
        age = str(stat.get('patient_age', 'N/A'))
        sex = str(stat.get('patient_sex', 'N/A'))
        modality = str(stat.get('modality', 'N/A')).upper()
        body = str(stat.get('body_part_examined', '')).upper()
        study_desc = str(stat.get('study_description', '')).strip()
        series_desc = str(stat.get('series_description', '')).strip()
        accession = str(stat.get('accession_number', '')).strip()
        study_id = str(stat.get('study_id', '')).strip()
        urgent = bool(stat.get('urgent', False))
        reasons = list(map(str, stat.get('urgent_reasons', []))) or []

        # PHI context (compute early to optionally show a headline badge)
        phi_flags: list[str] = []
        private_tags = []
        try:
            if isinstance(full, dict):
                phi_flags = list(map(str, full.get('phi_flags', [])))  # type: ignore[assignment]
                private_tags = full.get('private_tags', []) or []
            if not phi_flags:
                phi_flags = list(map(str, stat.get('phi_flags', [])))
        except Exception:
            pass

        # Date/time: try to pretty format and include relative time
        dt_display = str(stat.get('study_date_time', '')).strip()
        rel_text = ''
        date_str: str | None = None
        time_str: str | None = None
        if isinstance(full, dict):
            date_str = (full.get('study_date') or stat.get('study_date') or None)  # type: ignore[assignment]
            time_str = (full.get('study_time') or stat.get('study_time') or None)  # type: ignore[assignment]
        try:
            if date_str:
                fmt, future, dt_obj = format_dicom_datetime(date_str, time_str)  # type: ignore[misc]
                # We'll recompute relative text so we can control 'in' vs 'ago' formatting
                dt_display = fmt
                if dt_obj is not None:
                    now_ref = datetime.now()
                    rel = human_readable_delta(dt_obj, now_ref)
                    if rel and rel != 'just now':
                        rel_text = f" (in {rel})" if future else f" ({rel} ago)"
        except Exception:
            # Fallback to whatever stat provided
            pass

        # One-line headline similar to the screenshot style
        # Badge text kept compatible with tests: [URGENT] or [OK] (green)
        status = cc('[URGENT]', 'red', ['bold']) if urgent else cc('[OK]', 'green', ['bold'])
        age_sex = f"{cc(age, 'yellow', ['bold'])} {cc('|', 'white')} {cc(sex, 'white', ['bold'])}"
        mod_body = (
            f"{cc(modality, 'cyan', ['bold'])} {cc(body, 'cyan')}" if body else cc(modality, 'cyan', ['bold'])
        )
        # Avoid duplicating any existing relative suffix already embedded in dt_display
        base_dt = dt_display.split(' (')[0].strip() if dt_display else dt_display
        dt_part = f"{cc(base_dt, 'green')}{cc(rel_text, 'green') if rel_text else ''}"
        # Headline (no PHI badge here; PHI summary printed below)
        print(f"{status} {age_sex} — {mod_body} — {dt_part}")
        pad = '  '

        # Study and identifiers
        if study_desc or series_desc:
            print(f"{pad}{b('Study:')} {study_desc or '-'}")
            if series_desc:
                print(f"{pad}{b('Series:')} {series_desc}")

        ids_parts: list[str] = []
        if accession:
            ids_parts.append(f"ACC#{cc(accession, 'yellow', ['bold'])}")
        if study_id:
            ids_parts.append(f"ID:{cc(study_id, 'yellow')}")
        if ids_parts:
            print(f"{pad}{b('IDs:')} " + ", ".join(ids_parts))

        if urgent and reasons:
            print(f"{pad}{cc('Alert:', 'red', ['bold'])} {cc(', '.join(reasons), 'red')}")

        # Physicians (minimal block too) — summarize if available
        roles: list[str] = []
        if isinstance(full, dict):
            _ref = str(full.get('referring_physician_name', '')).strip()
            _att = (
                str(full.get('attending_physician_name', '')).strip()
                or str(full.get('attending_physicians_name', '')).strip()
            )
            _perf = str(full.get('performing_physician_name', '')).strip()
            _read = str(full.get('reading_physicians', '')).strip()
            _req = str(full.get('requesting_physician', '')).strip()
            _cons = str(full.get('consulting_physician_name', '')).strip()
            _appr = str(full.get('physician_approving_interpretation', '')).strip()
            _sched = str(full.get('scheduled_performing_physician_name', '')).strip()
            if _ref: roles.append(f"Ref: {_ref}")
            if _att: roles.append(f"Att: {_att}")
            if _perf: roles.append(f"Perf: {_perf}")
            if _read: roles.append(f"Read: {_read}")
            if _req: roles.append(f"Req: {_req}")
            if _cons: roles.append(f"Cons: {_cons}")
            if _appr: roles.append(f"Appr: {_appr}")
            if _sched: roles.append(f"Sched: {_sched}")
        if roles:
            print(f"{pad}{b('Physicians:')} " + " | ".join(roles))

        if phi_flags or private_tags:
            extras = []
            if phi_flags:
                extras.append(', '.join(phi_flags))
            if private_tags:
                try:
                    extras.append(f"Private tags: {len(private_tags)}")
                except Exception:
                    pass
            # New line and no indentation for PHI warning
            print()
            print(f"{cc('⚠ PHI Warning:', 'red', ['bold'])} {cc(' | '.join(extras), 'red')}")

        # Optional detailed section
        if detail and isinstance(full, dict):
            print()
            print(pad + cc('=== Detailed Information ===', 'cyan', ['bold']))
            station = str(full.get('station_name', '')).strip()
            if station:
                print(f"{pad}{b('Scanner:')} {cc(station, 'cyan')}")
            inst = str(full.get('institution_name', '')).strip()
            if inst:
                print(f"{pad}{b('Institution:')} {cc(inst, 'white')}")
            ref = str(full.get('referring_physician_name', '')).strip()
            attend = (
                str(full.get('attending_physician_name', '')).strip()
                or str(full.get('attending_physicians_name', '')).strip()
            )
            perf = str(full.get('performing_physician_name', '')).strip()
            read = str(full.get('reading_physicians', '')).strip()
            req = str(full.get('requesting_physician', '')).strip()
            cons = str(full.get('consulting_physician_name', '')).strip()
            appr = str(full.get('physician_approving_interpretation', '')).strip()
            sched = str(full.get('scheduled_performing_physician_name', '')).strip()

            role_parts: list[str] = []
            if ref:
                role_parts.append(f"Ref: {ref}")
            if attend:
                role_parts.append(f"Att: {attend}")
            if perf:
                role_parts.append(f"Perf: {perf}")
            if read:
                role_parts.append(f"Read: {read}")
            if req:
                role_parts.append(f"Req: {req}")
            if cons:
                role_parts.append(f"Cons: {cons}")
            if appr:
                role_parts.append(f"Appr: {appr}")
            if sched:
                role_parts.append(f"Sched: {sched}")

            if role_parts:
                print(f"{pad}{b('Physicians:')} {' | '.join(role_parts)}")

            # Referring contact details if present
            ref_addr = str(full.get('referring_physician_address', '')).strip()
            ref_tel = str(full.get('referring_physician_phone', '')).strip()
            if ref_addr or ref_tel:
                contact_bits: list[str] = []
                if ref_addr:
                    contact_bits.append(f"Addr: {ref_addr}")
                if ref_tel:
                    contact_bits.append(f"Tel: {ref_tel}")
                print(f"{pad}{b('Ref Contact:')} {' | '.join(contact_bits)}")
            manuf = str(full.get('manufacturer', '')).strip()
            model = str(full.get('manufacturer_model_name', '')).strip() or str(full.get('model', '')).strip()
            if manuf or model:
                mm = ' '.join([p for p in [manuf, model] if p])
                print(f"{pad}{b('Device:')} {cc(mm, 'white')}")
            mfs = str(full.get('magnetic_field_strength', '')).strip()
            if mfs:
                print(f"{pad}{b('Magnet:')} {cc(mfs, 'white')} {cc('T', 'white') if str(mfs).isdigit() else ''}")
            sw = str(full.get('software_versions', '')).strip()
            if sw:
                print(f"{pad}{b('Software:')} {cc(sw, 'white')}")

            # Patient & Study section (attractive two-column)
            pt_map = {
                'Patient Name': 'patient_name',
                'Patient ID': 'patient_id',
                'Birth Date': 'patient_birth_date',
                'Birth Time': 'patient_birth_time',
                'Age': 'patient_age',
                'Address': 'patient_address',
                'Other IDs': 'other_patient_ids',
                'Other Names': 'other_patient_names',
                'Accession': 'accession_number',
                'Study ID': 'study_id',
                'Series Desc': 'series_description',
                'Study Comments': 'study_comments',
                'Institution': 'institution_name',
                'Station': 'station_name',
                'Operators': 'operators_name',
            }
            def _fmt_pn(s: object) -> str:
                t = str(s)
                if '^' in t:
                    parts = t.split('^')
                    # Prefer First Last when possible
                    last = parts[0].strip()
                    first = (parts[1] if len(parts) > 1 else '').strip()
                    mid = (parts[2] if len(parts) > 2 else '').strip()
                    return ' '.join([p for p in [first, mid, last] if p]) or t
                return t
            pt_pairs = []
            for label, key in pt_map.items():
                v = full.get(key)
                if v not in (None, '', 'N/A'):
                    sv = _fmt_pn(v) if 'Name' in label else str(v)
                    pt_pairs.append((label, sv))
            if pt_pairs:
                print()
                print(pad + cc('--- Patient & Study ---', 'cyan', ['bold']))
                two = []
                def fmt_pair2(k: str, v: str) -> str:
                    return f"{cc('•', 'white')} {cc(k + ':', 'cyan')} {cc(v, 'white')}"
                for pair in pt_pairs:
                    two.append(pair)
                    if len(two) == 2:
                        left = fmt_pair2(*two[0])
                        right = fmt_pair2(*two[1])
                        print(pad + '  ' + left.ljust(60) + '  ' + right)
                        two = []
                if two:
                    print(pad + '  ' + fmt_pair2(*two[0]))
            rows = full.get('rows', 'N/A')
            cols = full.get('columns', 'N/A')
            bits = full.get('bits_allocated', 'N/A')
            series_no = full.get('series_number', '1')
            acq_no = full.get('acquisition_number', '1')
            print(f"{pad}{b('Image:')} {rows}×{cols}px, {bits}-bit, Series #{series_no}, Acq #{acq_no}")
            px = str(full.get('pixel_spacing', '') or '').strip()
            thick = str(full.get('slice_thickness', '') or '').strip()
            if px or thick:
                pieces = []
                if px:
                    pieces.append(f"Pixel spacing: {px}")
                if thick:
                    pieces.append(f"Slice thickness: {thick}")
                print(f"{pad}{b('Geometry:')} {' | '.join(pieces)}")

            tr = full.get('repetition_time', '')
            te = full.get('echo_time', '')
            seq_parts: list[str] = []
            if tr:
                seq_parts.append(f"TR={tr}ms")
            if te:
                seq_parts.append(f"TE={te}ms")
            if seq_parts:
                print(f"{pad}{b('Sequence:')} {' | '.join(seq_parts)}")

            # Extra block: Important DICOM tags
            important_map = {
                'BitsAllocated': 'bits_allocated',
                'BitsStored': 'bits_stored',
                'HighBit': 'high_bit',
                'SamplesPerPixel': 'samples_per_pixel',
                'PixelRepresentation': 'pixel_representation',
                'PlanarConfiguration': 'planar_configuration',
                'RescaleIntercept': 'rescale_intercept',
                'RescaleSlope': 'rescale_slope',
                'RescaleType': 'rescale_type',
                'PixelSpacing': 'pixel_spacing',
                'Rows': 'rows',
                'Columns': 'columns',
                'SliceThickness': 'slice_thickness',
                'SpacingBetweenSlices': 'spacing_between_slices',
                'ImageOrientationPatient': 'image_orientation_patient',
                'ImagePositionPatient': 'image_position_patient',
                'InstanceNumber': 'instance_number',
                'FrameOfReferenceUID': 'frame_of_reference_uid',
                'SOPClassUID': 'sop_class_uid',
                'SOPInstanceUID': 'sop_instance_uid',
                'StudyInstanceUID': 'study_instance_uid',
                'TransferSyntaxUID': 'transfer_syntax_uid',
            }
            def _short(v: object) -> str:
                s = str(v)
                # Keep columns tidy: aggressively shorten long values (e.g., UIDs)
                if len(s) > 36:
                    return s[:36] + '…'
                return s

            present_pairs: list[tuple[str, str]] = []
            for label, key in important_map.items():
                try:
                    val = full.get(key)
                except Exception:
                    val = None
                if val not in (None, '', 'N/A'):
                    present_pairs.append((label, _short(val)))
            if present_pairs:
                print()
                print(pad + cc('--- Important DICOM Tags ---', 'cyan', ['bold']))
                # Two-column bullet layout
                col = 2
                row: list[tuple[str, str]] = []
                def fmt_pair(k: str, v: str) -> str:
                    return f"{cc('•', 'white')} {cc(k + ':', 'cyan')} {cc(v, 'white')}"
                for pair in present_pairs:
                    row.append(pair)
                    if len(row) == col:
                        left = fmt_pair(*row[0])
                        right = fmt_pair(*row[1])
                        print(pad + '  ' + left.ljust(60) + '  ' + right)
                        row = []
                if row:
                    print(pad + '  ' + fmt_pair(*row[0]))

            phi_tags = full.get('phi_tags', [])
            if phi_tags:
                print()
                print(f"{cc('⚠ PHI Warning:', 'red', ['bold'])} {cc(', '.join(map(str, phi_tags)), 'red')}")

        # Don't spam INFO logs with duplicative STAT; use debug for traceability
        logging.debug('STAT: %s | %s | %s %s | %s', age, sex, modality, body, dt_display)

    except Exception as e:
        logging.error("Error formatting STAT output: %s", e)
        logging.debug(traceback.format_exc())


# --------------------------- dependency checking --------------------------


def check_dependencies() -> dict[str, bool]:
    deps = {
        'pillow': _PIL_AVAILABLE,
        'numpy': _PIL_AVAILABLE and 'numpy' in sys.modules,
        'pandas': _PANDAS_AVAILABLE,
        'tqdm': _TQDM,
        'dateutil.relativedelta': _RELATIVEDELTA_AVAILABLE,
        'cryptography': _CRYPTO_AVAILABLE,
    }
    return deps


# ----------------------------- utility funcs -----------------------------


def md5_short(s: str, n: int = 8) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:n]


def sanitize_for_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode('utf-8', errors='ignore')
        except Exception:
            return str(obj)
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    try:
        from pydicom.dataset import Dataset

        if isinstance(obj, Dataset):
            out = {}
            for k in obj.keys():
                try:
                    v = obj.get(k)
                except Exception:
                    v = None
                out[str(k)] = sanitize_for_json(v)
            return out
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def flatten_for_csv_row(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


# ------------------------ date/time humanization -------------------------


def parse_dicom_time_str(t: str | None) -> datetime.time | None:
    if not t:
        return None
    t = str(t).split('.')[0].ljust(6, '0')[:6]
    try:
        return datetime.strptime(t, "%H%M%S").time()
    except Exception:
        return None


def detailed_delta_components(
    dt_obj: datetime, now: datetime | None = None
) -> tuple[int, int, int, int, int, int]:
    if now is None:
        now = datetime.now()
    if _RELATIVEDELTA_AVAILABLE:
        rd = relativedelta(now, dt_obj) if now >= dt_obj else relativedelta(dt_obj, now)
        return (
            abs(rd.years),
            abs(rd.months),
            abs(rd.days),
            abs(rd.hours),
            abs(rd.minutes),
            abs(rd.seconds),
        )
    total_seconds = int(abs(int((now - dt_obj).total_seconds())))
    days = total_seconds // 86400
    years = days // 365
    days -= years * 365
    months = days // 30
    days -= months * 30
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return years, months, days, hours, minutes, seconds


def human_readable_delta(dt_obj: datetime, now: datetime | None = None) -> str:
    years, months, days, hours, minutes, seconds = detailed_delta_components(dt_obj, now)
    parts: list[str] = []
    if years:
        parts.append(f"{years} year{'s' if years!=1 else ''}")
    if months:
        parts.append(f"{months} month{'s' if months!=1 else ''}")
    if days:
        parts.append(f"{days} day{'s' if days!=1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours!=1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes!=1 else ''}")
    if not parts:
        return "just now"
    return ", ".join(parts[:4])


def format_dicom_datetime(
    date_str: str | None, time_str: str | None
) -> tuple[str, bool, datetime | None]:
    if not date_str:
        return "N/A", False, None
    try:
        date_obj = datetime.strptime(str(date_str), "%Y%m%d")
    except Exception:
        return "Invalid Date", False, None
    tp = parse_dicom_time_str(time_str)
    dt_obj = datetime.combine(date_obj.date(), tp) if tp else date_obj
    now = datetime.now()
    delta_seconds = int((now - dt_obj).total_seconds())
    future = delta_seconds < 0
    rel = human_readable_delta(dt_obj, now)
    rel_text = (
        f"in {rel}"
        if future and rel != "just now"
        else (
            f"{rel} ago"
            if not future and rel != "just now"
            else ("in a moment" if future else "just now")
        )
    )
    formatted = dt_obj.strftime("%d %B %Y, %I:%M %p")
    return f"{formatted} ({rel_text})", future, dt_obj


# ----------------------------- PHI, private tags -------------------------


def check_phi(ds: pydicom.dataset.Dataset) -> list[str]:
    phi: list[str] = []
    for k in [
        "PatientName",
        "PatientID",
        "PatientAddress",
        "OtherPatientIDsSequence",
        "ReferringPhysicianName",
        "StudyComments",
        "InstitutionName",
        "StationName",
    ]:
        if ds.get(k):
            phi.append(k)
    private = [t for t in ds.keys() if t.is_private]
    if private:
        phi.append(f"Private tags: {len(private)}")
    for seq in ["RequestingService", "RequestingPhysician", "RequestingPhysicianName"]:
        if ds.get(seq):
            phi.append(seq)
    return phi


def list_private_tags(ds: DicomDataset, show_values: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tags = sorted([t for t in ds.keys() if t.is_private], key=lambda x: (x.group, x.elem))
    for tag in tags:
        try:
            elem = ds[tag]
            tag_str = f"({tag.group:04x},{tag.elem:04x})"
            keyword = getattr(elem, 'keyword', '') or ''
            name = getattr(elem, 'name', '') or ''
            creator_tag = pydicom.tag.Tag(tag.group, 0x0010)
            creator = ds.get(creator_tag)
            creator_str = str(creator) if creator else ''
            value_preview = sanitize_for_json(elem.value)
            if isinstance(value_preview, str) and len(value_preview) > 200:
                vp = value_preview[:197] + "..."
            else:
                vp = value_preview
            full = sanitize_for_json(elem.value) if show_values else None
            out.append(
                {
                    'tag': tag_str,
                    'group': f"0x{tag.group:04x}",
                    'element': f"0x{tag.elem:04x}",
                    'keyword': keyword,
                    'name': name,
                    'creator': creator_str,
                    'value_preview': vp,
                    'full_value': full,
                }
            )
        except Exception:
            continue
    return out


# --------------------------- anonymization -------------------------------


def _pbkdf2_pseudonym(value: str, salt: bytes, iterations: int = 100000, length: int = 16) -> str:
    # Use PBKDF2 output directly, base64-encode for safe text
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=length, salt=salt, iterations=iterations)
    key = kdf.derive(value.encode('utf-8'))
    token = base64.urlsafe_b64encode(key).decode('utf-8')[:16]
    return "anon_" + token


def _hmac_pseudonym(value: str, salt: bytes) -> str:
    """Generate a pseudonym using HMAC with provided or default salt."""
    salt_bytes = salt or b'default_salt'
    value_bytes = str(value).encode('utf-8')

    # Generate HMAC of value using salt
    hash_fn = hashlib.sha256
    hm = hmac.new(salt_bytes, value_bytes, hash_fn).hexdigest()[:16]

    # Prefix with "anon_" to make it clear this is a pseudonym
    return f"anon_{hm}"


def pseudonymize_value(value: Any, salt: str | None = None) -> tuple[str | None, str]:
    """
    Returns (pseudonym, salt_hex_used).
    If salt was None, a random salt (bytes) is generated and its hex returned (but not logged verbatim).
    """
    if value is None:
        return None, (salt or '')
    v = str(value)
    if salt is None:
        salt_bytes = secrets.token_bytes(8)
        salt_hex = salt_bytes.hex()
        # Do NOT log the generated salt in full for security; we store it in the mapping only.
        logging.debug("Generated ephemeral salt for pseudonymization (not displayed)")
    else:
        salt_hex = salt
        salt_bytes = salt.encode('utf-8')
        if _CRYPTO_AVAILABLE and salt and len(salt) < 8:
            logging.warning(
                'Provided salt is short (<8 chars); consider longer salt for better security'
            )
    try:
        if _CRYPTO_AVAILABLE:
            pseudo = _pbkdf2_pseudonym(v, salt_bytes)
        else:
            pseudo = _hmac_pseudonym(v, salt_bytes)
            logging.warning(
                "cryptography not available — using HMAC fallback for pseudonymization (weaker). Install 'cryptography' for PBKDF2HMAC."
            )
    except Exception as e:
        logging.error("Pseudonymization failed for value %s: %s", v, e)
        pseudo = _hmac_pseudonym(v, salt_bytes if salt_bytes else b'default_salt')
    return pseudo, salt_hex


def apply_anonymization_to_sanitized(
    sanitized: dict[str, Any],
    tags: list[str],
    mode: str,
    salt: str | None,
    run_mapping: dict[str, Any],
    file_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Modifies sanitized dict in-place. Returns sanitized and the per-file mapping.
    run_mapping is updated with run_mapping[file_path] = per_file_map
    """
    per_file_map: dict[str, Any] = {}
    for tag in tags:
        for key in list(sanitized.keys()):
            if key.lower() == tag.lower():
                orig = sanitized.get(key)
                if mode == 'pseudonymize':
                    pseud, salt_used = pseudonymize_value(orig, salt)
                    sanitized[key] = pseud
                    per_file_map[key] = {'original': orig, 'pseudonym': pseud, 'salt': salt_used}
                else:
                    sanitized[key] = 'REDACTED'
                    per_file_map[key] = {'original': orig, 'redacted': True}
    if per_file_map:
        # Protect concurrent updates to the shared run_mapping
        try:
            with ANON_MAP_LOCK:
                run_mapping.setdefault(str(file_path), {}).update(per_file_map)
        except Exception:
            # Fallback: try naive update (best-effort) and log
            try:
                run_mapping.setdefault(str(file_path), {}).update(per_file_map)
            except Exception:
                logging.exception('Failed to update anonymization map for %s', file_path)
    return sanitized, per_file_map


# ----------------------- pixel extraction & thumbnails -------------------


def save_pixel_images(ds: DicomDataset, out_prefix: str, ext: str = '.png') -> list[str]:
    saved = []
    if 'PixelData' not in ds:
        return saved
    if not _PIL_AVAILABLE:
        logging.debug('Pillow/numpy not available; cannot save pixel images')
        return saved
    try:
        arr = ds.pixel_array
    except Exception as e:
        logging.debug('pixel_array decode failed: %s', e)
        return saved
    try:
        np_arr = np.asarray(arr)
    except Exception:
        return saved

    if np_arr.ndim == 2:
        frames = [np_arr]
    elif np_arr.ndim == 3:
        # Check if shape makes sense for sequential frames
        dims_ok = (
            np_arr.shape[0] <= 64  # Reasonable number of frames
            and np_arr.shape[1] > 4  # Non-trivial dimensions
            and np_arr.shape[2] > 4
        )
        if dims_ok:
            frames = [np_arr[i] for i in range(np_arr.shape[0])]
        else:
            frames = [np_arr]
    elif np_arr.ndim == 4:
        frames = [np_arr[i] for i in range(np_arr.shape[0])]
    else:
        frames = [np_arr]

    # Normalize extension format and get corresponding PIL format
    ext_l = ext.lower() if ext.startswith('.') else f".{ext.lower()}"

    format_map = {
        '.png': 'PNG',
        '.jpg': 'JPEG',
        '.jpeg': 'JPEG',
        '.bmp': 'BMP',
        '.tiff': 'TIFF',
        '.tif': 'TIFF',
    }
    pil_format = format_map.get(ext_l, 'PNG')

    for idx, frame in enumerate(frames):
        f = frame
        if getattr(f, 'dtype', None) is not None and f.dtype != np.uint8:
            try:
                fmin, fmax = float(f.min()), float(f.max())
                if fmax - fmin > 0:
                    f = ((f - fmin) / (fmax - fmin) * 255.0).astype(np.uint8)
                else:
                    f = (f * 0).astype(np.uint8)
            except Exception:
                try:
                    f = f.astype(np.uint8)
                except Exception:
                    continue
        try:
            img = Image.fromarray(f)
        except Exception:
            try:
                img = Image.fromarray(np.squeeze(f))
            except Exception:
                continue

        if img.mode not in ('L', 'RGB', 'RGBA'):
            try:
                img = img.convert('L')
            except Exception:
                img = img.convert('RGB')

        if len(frames) == 1:
            outpath = f"{out_prefix}{ext_l}"
        else:
            outpath = f"{out_prefix}_frame{idx}{ext_l}"

        try:
            Path(outpath).parent.mkdir(parents=True, exist_ok=True)
            img.save(outpath, format=pil_format)
            saved.append(str(outpath))
        except Exception:
            try:
                img.save(outpath)
                saved.append(str(outpath))
            except Exception:
                continue
    return saved


def try_thumbnail(ds: pydicom.dataset.Dataset, out_path: str, max_size: int = 256) -> bool:
    if not _PIL_AVAILABLE:
        return False
    try:
        if 'PixelData' not in ds:
            return False
        arr = ds.pixel_array
        if arr is None:
            return False
        np_arr = np.asarray(arr)
        if np_arr.ndim == 3:
            idx = np_arr.shape[0] // 2
            frame = np_arr[idx]
        elif np_arr.ndim == 4:
            frame = np_arr[0, ...]
        else:
            frame = np_arr
        if frame.dtype != np.uint8:
            fmin, fmax = float(frame.min()), float(frame.max())
            if fmax - fmin > 0:
                frame = ((frame - fmin) / (fmax - fmin) * 255.0).astype(np.uint8)
            else:
                frame = (frame * 0).astype(np.uint8)
        img = Image.fromarray(frame)
        img = img.convert('L') if img.mode != 'L' else img
        img.thumbnail((max_size, max_size))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format='PNG')
        return True
    except Exception as e:
        logging.debug('Thumbnail creation failed: %s', e)
        return False


# -------------------------- metadata extraction --------------------------


def compute_age_from_ds(ds: pydicom.dataset.Dataset) -> str:
    age = ds.get("PatientAge")
    if age and str(age).strip():
        return str(age)
    bdate = ds.get("PatientBirthDate") or ds.get("PatientsBirthDate")
    if bdate:
        try:
            bd = datetime.strptime(str(bdate), "%Y%m%d")
            years = (datetime.now() - bd).days // 365
            return f"{years}Y"
        except Exception:
            return "N/A"
    return "N/A"


def is_urgent(ds: DicomDataset) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    mod = str(ds.get("Modality", "")).upper()
    desc = str(ds.get("StudyDescription", "")).upper()
    if mod in ("CT", "MR") and any(
        x in desc
        for x in ["BRAIN", "HEAD", "STROKE", "TRAUMA", "INTRACRANIAL", "ICH", "HEMORRHAGE"]
    ):
        reasons.append("Head study with stroke/trauma keywords")
    if "ANGIO" in desc or "CTA" in desc or "CT ANGIO" in desc:
        reasons.append("Angio/CTA study")
    if mod == "US" and "FAST" in desc:
        reasons.append("FAST ultrasound")
    age = compute_age_from_ds(ds)
    try:
        if isinstance(age, str) and age.endswith('Y'):
            a = int(age.rstrip('Y'))
            if a >= 65 and mod in ("CT", "MR") and "BRAIN" in desc:
                reasons.append("Elderly patient + brain imaging")
    except Exception:
        pass
    return (len(reasons) > 0, reasons)


def get_dicom_metadata_from_ds(ds: DicomDataset, file_path: str) -> dict[str, Any]:
    """Extract metadata from a DICOM dataset into a structured dictionary."""
    try:
        study_dt_str, warn_future, dt_obj = format_dicom_datetime(
            ds.get('StudyDate', 'N/A'), ds.get('StudyTime', 'N/A')
        )
        stat_report = {
            "patient_id": str(ds.get('PatientID', 'N/A')),
            "patient_name": str(ds.get('PatientName', 'N/A')),
            "patient_age": compute_age_from_ds(ds),
            "patient_sex": str(ds.get('PatientSex', 'N/A')),
            "modality": str(ds.get('Modality', 'N/A')),
            "body_part_examined": str(ds.get('BodyPartExamined', 'N/A')),
            "study_description": str(ds.get('StudyDescription', 'N/A')),
            "study_date_time": study_dt_str,
            "phi_removed": ds.get('PatientIdentityRemoved', 'Unknown'),
            "_study_date_future": warn_future,
            "_file_path": file_path,
        }
        full_report = {
            "manufacturer": str(ds.get('Manufacturer', 'N/A')),
            "manufacturer_model_name": str(ds.get('ManufacturerModelName', ds.get('ManufacturerModelName', 'N/A'))),
            "model": str(ds.get('ManufacturerModelName', 'N/A')),
            "software_versions": ds.get('SoftwareVersions', 'N/A'),
            "magnetic_field_strength": ds.get('MagneticFieldStrength', 'N/A'),
            "slice_thickness": ds.get('SliceThickness', 'N/A'),
            "pixel_spacing": ds.get('PixelSpacing', 'N/A'),
            "rows": ds.get('Rows', 'N/A'),
            "columns": ds.get('Columns', 'N/A'),
            "bits_allocated": ds.get('BitsAllocated', 'N/A'),
            "photometric_interpretation": ds.get('PhotometricInterpretation', 'N/A'),
            "study_instance_uid": ds.get('StudyInstanceUID', 'N/A'),
            "series_instance_uid": ds.get('SeriesInstanceUID', 'N/A'),
            "transfer_syntax_uid": str(getattr(ds.file_meta, 'TransferSyntaxUID', 'N/A')),
            "number_of_frames": getattr(ds, 'NumberOfFrames', 'N/A'),
            "accession_number": ds.get('AccessionNumber', 'N/A'),
            "study_id": ds.get('StudyID', 'N/A'),
            # sequence/series
            "series_number": ds.get('SeriesNumber', 'N/A'),
            "acquisition_number": ds.get('AcquisitionNumber', 'N/A'),
            "repetition_time": ds.get('RepetitionTime', ''),
            "echo_time": ds.get('EchoTime', ''),
            # time breakdown for nicer formatting
            "study_date": ds.get('StudyDate', None),
            "study_time": ds.get('StudyTime', None),
            # site/device/personnel
            "station_name": ds.get('StationName', ''),
            "institution_name": ds.get('InstitutionName', ''),
            "referring_physician_name": ds.get('ReferringPhysicianName', ''),
            "referring_physician_address": ds.get('ReferringPhysicianAddress', ''),
            "referring_physician_phone": ds.get('ReferringPhysicianTelephoneNumbers', ''),
            "referring_physician_identification_sequence": ds.get('ReferringPhysicianIdentificationSequence', None),
            "performing_physician_name": ds.get('PerformingPhysicianName', ''),
            "performing_physician_identification_sequence": ds.get('PerformingPhysicianIdentificationSequence', None),
            "physicians_of_record": ds.get('PhysiciansOfRecord', ''),
            "physicians_of_record_identification_sequence": ds.get('PhysiciansOfRecordIdentificationSequence', None),
            "reading_physicians": ds.get('NameOfPhysiciansReadingStudy', ''),
            "reading_physicians_identification_sequence": ds.get('PhysiciansReadingStudyIdentificationSequence', None),
            "requesting_physician": ds.get('RequestingPhysician', ''),
            "requesting_physician_identification_sequence": ds.get('RequestingPhysicianIdentificationSequence', None),
            "consulting_physician_name": ds.get('ConsultingPhysicianName', ''),
            "consulting_physician_identification_sequence": ds.get('ConsultingPhysicianIdentificationSequence', None),
            "scheduled_performing_physician_name": ds.get("ScheduledPerformingPhysicianName", ''),
            "scheduled_performing_physician_identification_sequence": ds.get('ScheduledPerformingPhysicianIdentificationSequence', None),
            "physician_approving_interpretation": ds.get('PhysicianApprovingInterpretation', ''),
            # heuristic for attending
            "attending_physician_name": ds.get('AttendingPhysicianName', ds.get('PerformingPhysicianName', '')),
            # patient & study extras (snake_case)
            "patient_name": ds.get('PatientName', ''),
            "patient_id": ds.get('PatientID', ''),
            "patient_birth_date": ds.get('PatientBirthDate', ''),
            "patient_birth_time": ds.get('PatientBirthTime', ''),
            "patient_address": ds.get('PatientAddress', ''),
            "other_patient_ids": ds.get('OtherPatientIDs', ''),
            "other_patient_ids_sequence": ds.get('OtherPatientIDsSequence', None),
            "other_patient_names": ds.get('OtherPatientNames', ''),
            "operators_name": ds.get('OperatorsName', ''),
            "station_name": ds.get('StationName', ''),
            "institution_name": ds.get('InstitutionName', ''),
            "accession_number": ds.get('AccessionNumber', ''),
            "study_id": ds.get('StudyID', ''),
            "series_description": ds.get('SeriesDescription', ''),
            "study_comments": ds.get('StudyComments', ''),
        }
        phi_flags = check_phi(ds)
        urgent, reasons = is_urgent(ds)
        private_tags = list_private_tags(ds, show_values=False)
        return {
            'stat_report': stat_report,
            'full_report': full_report,
            'phi_flags': phi_flags,
            'urgent': urgent,
            'urgent_reasons': reasons,
            'private_tags': private_tags,
        }
    except Exception as e:
        return {'Error': f"Unexpected error extracting metadata: {e}\n{traceback.format_exc()}"}


# ----------------------- outputs & saving helpers ------------------------


def parse_output_items(items: list[str], outdir: str) -> dict[str, list[str]]:
    outmap: dict[str, list[str]] = {}
    outdir = Path(outdir)
    for raw in items:
        if not raw:
            continue
        parts = [p.strip() for p in raw.replace(';', ',').split(',') if p.strip()]
        for p in parts:
            if '=' in p:
                t, fn = p.split('=', 1)
                t = t.strip().lower()
                if t not in SUPPORTED_TYPES:
                    logging.warning("Unsupported output type '%s' in '%s' -> skipped", t, p)
                    continue
                path = (
                    Path(fn)
                    if Path(fn).is_absolute() or Path(fn).parent != Path('.')
                    else Path.cwd() / fn
                )
                outmap.setdefault(t, []).append(str(path))
                continue
            if p.lower() in SUPPORTED_TYPES:
                outmap.setdefault(p.lower(), []).append('')
                continue
            if '.' in p:
                ext = p.rsplit('.', 1)[1].lower()
                t = EXT_TO_TYPE.get(ext)
                if not t:
                    logging.warning(
                        "Unknown extension '.%s' for '%s' -- supported: .json .csv .html .png .jpg .jpeg .bmp .tiff",
                        ext,
                        p,
                    )
                    continue
                path = (
                    Path(p)
                    if Path(p).is_absolute() or Path(p).parent != Path('.')
                    else Path.cwd() / p
                )
                outmap.setdefault(t, []).append(str(path))
                continue
            logging.warning(
                "Unrecognized output argument '%s' -- supported types: %s",
                p,
                ','.join(sorted(SUPPORTED_TYPES)),
            )
    return outmap


# -------------------------- find files (pathlib) -------------------------


def find_dicom_files(root: str, max_depth: int | None = None) -> list[str]:
    rootp = Path(root)
    if not rootp.exists():
        return []
    exts = {'.dcm', '.ima', '.img'}
    if max_depth is None:
        return [str(p) for p in rootp.rglob("*") if p.suffix.lower() in exts]
    else:
        out: list[str] = []
        base_level = len(rootp.parts)
        for p in rootp.rglob("*"):
            if p.suffix.lower() not in exts:
                continue
            if len(p.parts) - base_level <= max_depth:
                out.append(str(p))
        return out


# ----------------------- streaming aggregation helpers -------------------


def stream_write_csv(rows_iter: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Stream write dictionary rows to a CSV file.

    Each row is flattened before writing to handle nested structures.
    Header is written from first row's keys.
    """
    first = True
    flushed = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = None
        for i, row in enumerate(rows_iter):
            flat_row = {k: flatten_for_csv_row(v) for k, v in row.items()}
            if first:
                writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
                writer.writeheader()
                first = False
            writer.writerow(flat_row)
            flushed += 1
            if flushed % 1000 == 0:
                f.flush()
    logging.info("Streamed CSV -> %s", output_path)


def stream_write_json(rows_iter: Iterable[dict[str, Any]], output_path: Path):
    first = True
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[')
        for row in rows_iter:
            if not first:
                f.write(',\n')
            f.write(json.dumps(row, ensure_ascii=False))
            first = False
        f.write(']')
    logging.info("Streamed JSON -> %s", output_path)


# ------------------------------- processing --------------------------------


def process_and_save(
    path: str,
    args: argparse.Namespace,
    outputs_map: dict[str, list[str]],
    run_mapping: dict[str, Any],
    dry_run: bool = False,
    suppress_details: bool = False,
) -> dict[str, Any] | None:
    pathp = Path(path)
    try:
        ds = pydicom.dcmread(str(pathp), force=args.force)
    except InvalidDicomError:
        logging.error("Not a valid DICOM: %s", path)
        return None
    except Exception as e:
        logging.error("Failed to read DICOM %s: %s", path, e)
        logging.debug(traceback.format_exc())
        return None

    metadata = get_dicom_metadata_from_ds(ds, str(pathp))

    if 'Error' in metadata:
        logging.error("Metadata error for %s: %s", path, metadata.get('Error'))
        return None

    sanitized: dict[str, Any] = {}
    stat = metadata.get('stat_report', {})
    full = metadata.get('full_report', {})
    sanitized.update(
        {k: sanitize_for_json(v) for k, v in stat.items() if not str(k).startswith('_')}
    )
    sanitized.update({k: sanitize_for_json(v) for k, v in full.items()})
    sanitized['phi_flags'] = metadata.get('phi_flags', [])
    sanitized['urgent'] = metadata.get('urgent', False)
    sanitized['urgent_reasons'] = metadata.get('urgent_reasons', [])
    sanitized['private_tags'] = metadata.get('private_tags', [])

    per_file_map = {}
    if args.anonymize:
        if not args.anonymize_tags:
            tags_to_anon = DEFAULT_ANON_TAGS
        else:
            tags_to_anon = [t.strip() for t in args.anonymize_tags.split(',') if t.strip()]
        sanitized, per_file_map = apply_anonymization_to_sanitized(
            sanitized,
            tags_to_anon,
            args.anonymize_mode,
            args.anonymize_salt,
            run_mapping,
            str(pathp.resolve()),
        )

    if not args.batch or args.verbose > 0:
        # If user requested the minimal STAT view, render a pretty colored single-line summary
        show_stat = getattr(args, 'minimal', False) and not getattr(args, 'quiet', False)
        if show_stat:
            try:
                # Pass both the stat summary and full metadata for detailed views
                pretty_print_stat(
                    stat=sanitized,
                    full=metadata.get('full_report', {}),
                    color=not getattr(args, 'quiet', False),
                    detail=getattr(args, 'detail', False) or getattr(args, 'full', False),
                )

                # Additional full-mode metadata dump
                if getattr(args, 'full', False):
                    print("\n=== Complete Metadata Dump ===")
                    print(json.dumps(metadata.get('full_report', {}), indent=2, sort_keys=True))
            except Exception as e:
                logging.error('Failed to show detailed view: %s', e)
                logging.debug(traceback.format_exc())
                # Fall back to basic info logging
                logging.info(
                    'STAT: %s | %s | %s %s | %s',
                    sanitized.get('patient_age', 'N/A'),
                    sanitized.get('patient_sex', 'N/A'),
                    sanitized.get('modality', 'N/A'),
                    sanitized.get('body_part_examined', 'N/A'),
                    sanitized.get('study_date_time', 'N/A'),
                )
        else:
            logging.info(
                'STAT: %s | %s | %s %s | %s',
                sanitized.get('patient_age', 'N/A'),
                sanitized.get('patient_sex', 'N/A'),
                sanitized.get('modality', 'N/A'),
                sanitized.get('body_part_examined', 'N/A'),
                sanitized.get('study_date_time', 'N/A'),
            )
    else:
        if not suppress_details:
            logging.info('Processed: %s', path)
        else:
            logging.debug('Processed (suppressed details): %s', path)

    base = pathp.stem
    uniq = md5_short(str(pathp.resolve()))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    if 'json' in outputs_map:
        targets = outputs_map.get('json') or ['']
        for t in targets:
            outpath = Path(t) if t else out_dir / f"{base}_{uniq}_metadata.json"
            if args.dry_run:
                logging.info('DRY RUN: would save JSON -> %s', outpath)
            else:
                try:
                    if outpath.exists() and args.no_overwrite:
                        logging.warning('Skipping existing file (no-overwrite): %s', outpath)
                    else:
                        outpath.parent.mkdir(parents=True, exist_ok=True)
                        with open(outpath, 'w', encoding='utf-8') as f:
                            json.dump(sanitized, f, indent=2, ensure_ascii=False)
                        logging.info('Saved JSON -> %s', outpath)
                except Exception as e:
                    logging.error('Failed to save JSON: %s', e)

    # CSV per-file
    if 'csv' in outputs_map:
        targets = outputs_map.get('csv') or ['']
        for t in targets:
            outpath = Path(t) if t else out_dir / f"{base}_{uniq}_metadata.csv"

            if args.dry_run:
                logging.info("DRY RUN: would save CSV -> %s", outpath)
                continue

            if outpath.exists() and args.no_overwrite:
                logging.warning("Skipping existing file (no-overwrite): %s", outpath)
                continue

            try:
                # Ensure parent dir exists
                outpath.parent.mkdir(parents=True, exist_ok=True)

                with open(outpath, 'w', newline='', encoding='utf-8') as f:
                    if not sanitized:
                        # Nothing to write — still create an empty CSV with no headers
                        writer = csv.DictWriter(f, fieldnames=[])
                        writer.writeheader()
                        logging.info("Saved CSV (empty) -> %s", outpath)
                        continue

                    # Prepare writer and header
                    fieldnames = list(sanitized.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    # Build a safe serialized row (avoid broken JSON dumps causing crash)
                    formatted_row = {}
                    for k, v in sanitized.items():
                        if isinstance(v, (dict, list)):
                            try:
                                formatted_row[k] = json.dumps(v, ensure_ascii=False)
                            except Exception:
                                # fallback to str() if JSON serialization fails
                                formatted_row[k] = str(v)
                        else:
                            formatted_row[k] = v

                    # Try writing the row and catch common I/O/CSV errors
                    try:
                        writer.writerow(formatted_row)
                        logging.info("Saved CSV -> %s", outpath)
                    except (OSError, csv.Error) as e:
                        logging.error("Failed to write CSV -> %s : %s", outpath, e, exc_info=True)
                    except Exception:
                        logging.exception("Unexpected error while writing CSV -> %s", outpath)

            except OSError as e:
                logging.error(
                    "Failed to prepare or open CSV file %s: %s", outpath, e, exc_info=True
                )

    # Thumbnail
    thumb_path = None
    if 'thumbnail' in outputs_map:
        targets = outputs_map.get('thumbnail') or ['']
        for t in targets:
            dest = Path(t) if t else out_dir / 'thumbnails' / f"{base}_{uniq}_thumb.png"
            if args.dry_run:
                logging.info('DRY RUN: would create thumbnail -> %s', dest)
            else:
                try:
                    ok = try_thumbnail(ds, str(dest))
                    if ok:
                        logging.info('Thumbnail -> %s', dest)
                        thumb_path = str(dest)
                    else:
                        logging.warning(
                            'Thumbnail unavailable or failed (compressed/unsupported) for %s', path
                        )
                except Exception as e:
                    logging.error('Thumbnail generation failed: %s', e)

    # HTML
    if 'html' in outputs_map:
        targets = outputs_map.get('html') or ['']
        for t in targets:
            dest = Path(t) if t else out_dir / f"{base}_{uniq}_report.html"
            if args.dry_run:
                logging.info('DRY RUN: would save HTML -> %s', dest)
            else:
                try:
                    generate_html_report(
                        {
                            'stat_report': stat,
                            'full_report': full,
                            'private_tags': sanitized.get('private_tags', []),
                            'urgent': sanitized.get('urgent', False),
                            'urgent_reasons': sanitized.get('urgent_reasons', []),
                            'phi_flags': sanitized.get('phi_flags', []),
                        },
                        thumb_path,
                        str(dest),
                    )
                    logging.info('HTML report -> %s', dest)
                except Exception as e:
                    logging.error('Failed to save HTML: %s', e)

    # FHIR
    if 'fhir' in outputs_map:
        targets = outputs_map.get('fhir') or ['']
        for t in targets:
            dest = Path(t) if t else out_dir / f"{base}_{uniq}_imagingstudy.json"
            if args.dry_run:
                logging.info('DRY RUN: would save FHIR -> %s', dest)
            else:
                try:
                    imaging_study = dicom_to_fhir_imagingstudy(sanitized)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest, 'w', encoding='utf-8') as f:
                        json.dump(imaging_study, f, indent=2, ensure_ascii=False)
                    logging.info('FHIR ImagingStudy -> %s', dest)
                except Exception as e:
                    logging.error('Failed to save FHIR JSON: %s', e)

    # IMAGE (pixel extraction)
    if 'image' in outputs_map:
        targets = outputs_map.get('image') or ['']
        for t in targets:
            if t and Path(t).suffix:
                outpath = (
                    Path(t)
                    if Path(t).is_absolute() or Path(t).parent != Path('.')
                    else Path.cwd() / t
                )
                prefix_no_ext = str(outpath.with_suffix(''))
                ext = outpath.suffix.lower() or '.png'
                allowed = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
                if ext not in allowed:
                    logging.warning("Unsupported image extension '%s' -> defaulting to .png", ext)
                    ext = '.png'
                    prefix_no_ext = str(outpath.with_suffix(''))
            else:
                prefix_no_ext = str(out_dir / f"{base}_{uniq}_image")
                ext = '.png'
            if args.dry_run:
                logging.info('DRY RUN: would save pixel images -> %s*%s', prefix_no_ext, ext)
            else:
                try:
                    saved = save_pixel_images(ds, prefix_no_ext, ext=ext)
                    if saved:
                        logging.info('Saved pixel image(s): %s', saved)
                    else:
                        logging.warning(
                            'No pixel images saved (pixel data missing, compressed or not decodable) for %s',
                            path,
                        )
                except Exception as e:
                    logging.error('Failed to extract/save pixel images: %s', e)

    # REPORT (metadata-as-image)
    if 'report' in outputs_map:
        targets = outputs_map.get('report') or ['']
        for t in targets:
            dest = Path(t) if t else out_dir / f"{base}_{uniq}_metadata_report.png"
            if args.dry_run:
                logging.info('DRY RUN: would generate metadata image -> %s', dest)
            else:
                try:
                    generate_metadata_image(
                        {
                            'stat_report': stat,
                            'full_report': full,
                            'private_tags': sanitized.get('private_tags', []),
                            'urgent': sanitized.get('urgent', False),
                            'urgent_reasons': sanitized.get('urgent_reasons', []),
                            'phi_flags': sanitized.get('phi_flags', []),
                        },
                        thumb_path,
                        str(dest),
                    )
                    logging.info('Metadata image -> %s', dest)
                except Exception as e:
                    logging.error('Failed to generate metadata image: %s', e)

    result_record = sanitized.copy()
    result_record['_file_path'] = str(pathp.resolve())
    if per_file_map:
        result_record['_anon_map'] = per_file_map
    # If user requested an anonymization map file for single-file runs, write it
    try:
        if args.anonymize and getattr(args, 'anonymize_map', None) and run_mapping:
            try:
                amap_path = Path(args.anonymize_map)
                amap_path.parent.mkdir(parents=True, exist_ok=True)
                with open(amap_path, 'w', encoding='utf-8') as f:
                    json.dump(run_mapping, f, indent=2, ensure_ascii=False)
                logging.info('Saved anonymization map -> %s', amap_path)
            except Exception as e:
                logging.error('Failed to save anonymization map: %s', e)
    except Exception:
        # defensive: do not let map-writing failures block normal return
        logging.debug('Anonymization map write check skipped or failed')
    return result_record


# -------------------------- report/image helpers -------------------------


def generate_html_report(metadata: dict[str, Any], thumbnail_path: str | None, out_html: str):
    # Expect normalized keys provided by process_and_save
    stat = metadata.get('stat_report', {})
    full = metadata.get('full_report', {})
    urgent = metadata.get('urgent', False)
    urgent_reasons = metadata.get('urgent_reasons', [])
    phi_flags = metadata.get('phi_flags', [])
    private = metadata.get('private_tags', [])
    html_lines: list[str] = []
    html_lines.append('<!doctype html>')
    html_lines.append('<html><head><meta charset="utf-8"><title>DICOM Report</title>')
    html_lines.append(
        '<style>body{font-family:Arial,Helvetica,sans-serif;padding:16px} .stat{background:#ffecec;padding:8px;border-radius:6px} .full{background:#eef9ec;padding:8px;border-radius:6px} h1{font-size:18px}</style>'
    )
    html_lines.append('</head><body>')
    html_lines.append('<h1>DICOM Metadata Report</h1>')
    if thumbnail_path and Path(thumbnail_path).exists():
        html_lines.append(
            f'<img src="{Path(thumbnail_path).name}" alt="thumbnail" style="max-width:200px;float:right;margin-left:12px">'
        )
    html_lines.append('<h2>STAT (critical)</h2>')
    html_lines.append('<div class="stat"><ul>')
    for k, v in stat.items():
        if k.startswith('_'):
            continue
        html_lines.append(f'<li><strong>{k}:</strong> {sanitize_for_json(v)}</li>')
    html_lines.append('</ul></div>')
    html_lines.append('<h2>Full (technical)</h2>')
    html_lines.append('<div class="full"><ul>')
    for k, v in full.items():
        html_lines.append(f'<li><strong>{k}:</strong> {sanitize_for_json(v)}</li>')
    html_lines.append('</ul></div>')
    html_lines.append('<h2>Private Tags</h2>')
    html_lines.append('<div class="full"><ul>')
    for p in private:
        html_lines.append(
            f'<li><strong>{p.get("tag")}:</strong> Creator: {p.get("creator")} | Name: {p.get("name")} | Preview: {p.get("value_preview")}</li>'
        )
    html_lines.append('</ul></div>')
    html_lines.append('</body></html>')
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_lines))


def generate_metadata_image(
    metadata: dict[str, Any],
    thumbnail_path: str | None,
    out_image: str,
    width: int = 1200,
    img: PILImage = None,
    draw: PILDraw = None,
    font: PILFont = None,
):
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow not installed")
    # normalized metadata keys
    stat = metadata.get('stat_report', {})
    full = metadata.get('full_report', {})
    private = metadata.get('private_tags', [])
    urgent = metadata.get('urgent', False)
    urgent_reasons = metadata.get('urgent_reasons', [])
    phi_flags = metadata.get('phi_flags', [])
    margin = 24
    right_col_width = 320 if thumbnail_path else 0
    lines: list[str] = []
    lines.append("DICOM METADATA REPORT")
    lines.append("")
    lines.append("STAT (critical):")
    for k, v in stat.items():
        if k.startswith('_'):
            continue
        lines.append(f"{k}: {v}")
    lines.append("")
    if urgent:
        lines.append(f"URGENT: {'; '.join(urgent_reasons)}")
        lines.append("")
    if phi_flags:
        lines.append(f"PHI-like: {', '.join(phi_flags)}")
        lines.append("")
    lines.append("Private Tags:")
    for p in private:
        lines.append(
            f"{p.get('tag')} {p.get('creator') or 'N/A'} {p.get('name') or p.get('keyword') or ''}"
        )
        lines.append(f"Preview: {p.get('value_preview')}")
    lines.append("")
    lines.append("Full (technical):")
    for k, v in full.items():
        lines.append(f"{k}: {v}")
    try:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    except Exception:
        font = None
        title_font = None
    line_h = 14
    if title_font:
        try:
            line_h = max(line_h, title_font.getsize("A")[1])
        except Exception:
            line_h = 14
    canvas_h = margin * 2 + line_h * (len(lines) + 2)
    img = Image.new('RGB', (width, max(canvas_h, 200)), color='white')
    draw = ImageDraw.Draw(img)
    x = margin
    y = margin
    if title_font:
        draw.text((x, y), "DICOM METADATA REPORT", fill='black', font=title_font)
    else:
        draw.text((x, y), "DICOM METADATA REPORT", fill='black')
    y += line_h * 2
    for line in lines:
        if len(line) > 120:
            for chunk in [line[i : i + 120] for i in range(0, len(line), 120)]:
                draw.text((x, y), chunk, fill='black', font=font)
                y += line_h
        else:
            draw.text((x, y), line, fill='black', font=font)
            y += line_h
    if thumbnail_path and Path(thumbnail_path).exists():
        try:
            thumb = Image.open(thumbnail_path)
            thumb.thumbnail((right_col_width, right_col_width))
            img.paste(thumb, (width - right_col_width - margin, margin))
        except Exception:
            pass
    Path(out_image).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_image, format='PNG')


# --------------------------- FHIR mapping helper -------------------------


def dicom_to_fhir_imagingstudy(sanitized: dict[str, Any]) -> dict[str, Any]:
    study = {
        "resourceType": "ImagingStudy",
        "identifier": [
            {"system": "urn:dicom:uid", "value": sanitized.get('study_instance_uid', '')}
        ],
        "status": "unknown",
        "modality": [
            {
                "system": "http://dicom.nema.org/resources/ontology/DCM",
                "code": sanitized.get('modality', ''),
            }
        ],
        "subject": {"display": sanitized.get('patient_name', '')},
        "started": sanitized.get('study_date_time', ''),
        "description": sanitized.get('study_description', ''),
        "numberOfSeries": 1,
        "numberOfInstances": 1,
        "series": [
            {
                "uid": sanitized.get('series_instance_uid', ''),
                "number": 1,
                "modality": {
                    "system": "http://dicom.nema.org/resources/ontology/DCM",
                    "code": sanitized.get('modality', ''),
                },
                "instance": [
                    {"uid": sanitized.get('study_instance_uid', ''), "sopClass": "", "number": 1}
                ],
            }
        ],
    }
    return study


# --------------------------------- Main ---------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print(f"dicomaster version {__version__}")
        return

    configure_logging(args.quiet, args.verbose, args.log_file)

    if args.check_deps:
        deps = check_dependencies()
        for k, v in deps.items():
            logging.info('%s: %s', k, 'OK' if v else 'MISSING')
        return

    show_banner(args)

    outputs_map = parse_output_items(args.output or [], args.output_dir)

    # Export schema quick path
    if args.export_schema:
        schema_file = (
            Path(args.export_schema)
            if isinstance(args.export_schema, str)
            else Path('dicom_schema.csv')
        )
        header = [
            'study_id',
            'series_id',
            'instance_id',
            'patient_id',
            'patient_age',
            'patient_sex',
            'modality',
            'body_part',
            'manufacturer',
            'model',
            'study_date',
            'study_time',
            'study_date_time',
            'path_to_image',
            'rows',
            'columns',
            'pixel_spacing',
            'urgent',
            'private_tags_count',
            'accession_number',
        ]
        logging.info('Writing schema header to %s', schema_file)
        schema_file.parent.mkdir(parents=True, exist_ok=True)
        with open(schema_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        return

    # Interactive REPL
    if args.path is None and not args.no_interactive:
        logging.info('Interactive mode. Enter DICOM path (or "exit").')
        while True:
            try:
                user_in = input('File path (or exit): ').strip()
            except EOFError:
                logging.info('Exiting REPL.')
                break
            if not user_in:
                continue
            if user_in.lower() in ('exit', 'quit', 'q'):
                logging.info('Bye — have a nice day!')
                break
            if not Path(user_in).exists():
                logging.error('Path not found: %s', user_in)
                continue
            run_map = {}
            res = process_and_save(
                user_in, args, outputs_map, run_map, dry_run=args.dry_run, suppress_details=False
            )
            if res:
                logging.info('Done for %s', user_in)
        return

    # Non-interactive
    if args.batch:
        if not args.path or not Path(args.path).is_dir():
            logging.error('Batch path is not a directory: %s', args.path)
            sys.exit(1)

        files = find_dicom_files(args.path, max_depth=args.max_depth)
        if not files:
            logging.warning('No DICOM files found under %s', args.path)
            return

        total_files = len(files)
        logging.info('Processing %d files', total_files)

        suppress_details = total_files >= max(1, args.min_progress_report) and args.verbose == 0

        cpu_hint = (os.cpu_count() or 4) * 2
        threads_cap = min(args.threads, max(1, cpu_hint), MAX_THREAD_CAP)
        if threads_cap != args.threads:
            logging.debug(
                'Thread cap applied: using %d threads (requested %d)', threads_cap, args.threads
            )

        # prepare agg targets
        agg_csv_path = None
        agg_json_path = None
        if 'agg-csv' in outputs_map:
            targets = outputs_map.get('agg-csv') or ['']
            agg_csv_path = (
                Path(targets[0]) if targets[0] else Path(args.output_dir) / 'combined_metadata.csv'
            )
            agg_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if 'agg-json' in outputs_map:
            targets = outputs_map.get('agg-json') or ['']
            agg_json_path = (
                Path(targets[0]) if targets[0] else Path(args.output_dir) / 'combined_metadata.json'
            )
            agg_json_path.parent.mkdir(parents=True, exist_ok=True)

        run_mapping: dict[str, Any] = {}

        executor = ThreadPoolExecutor(max_workers=max(1, threads_cap))
        futures = {
            executor.submit(
                process_and_save,
                fpath,
                args,
                outputs_map,
                run_mapping,
                args.dry_run,
                suppress_details,
            ): fpath
            for fpath in files
        }

        progress_interval = max(1, min(10, max(1, total_files // 10)))

        # streaming agg if no pandas
        if (agg_csv_path or agg_json_path) and not _PANDAS_AVAILABLE:

            def rows_generator():
                completed = 0
                for fut in as_completed(futures):
                    try:
                        res = fut.result(timeout=args.timeout)
                        if res:
                            completed += 1
                            res_copy = {k: v for k, v in res.items() if not k.startswith('_')}
                            yield res_copy
                        if completed % progress_interval == 0:
                            logging.info('Progress: %d / %d', completed, total_files)
                    except Exception as e:
                        logging.error('Error processing file: %s', e)
                        logging.debug(traceback.format_exc())
                        continue

            if agg_csv_path:
                if args.dry_run:
                    logging.info('DRY RUN: would stream-aggregate CSV -> %s', agg_csv_path)
                else:
                    stream_write_csv(rows_generator(), agg_csv_path)
            if agg_json_path:
                if args.dry_run:
                    logging.info('DRY RUN: would stream-aggregate JSON -> %s', agg_json_path)
                else:
                    stream_write_json(rows_generator(), agg_json_path)
        else:
            results: list[dict[str, Any]] = []
            completed = 0
            use_tqdm = _TQDM and not args.quiet
            if use_tqdm:
                for fut in tqdm(as_completed(futures), total=len(futures), desc='Processing'):
                    try:
                        res = fut.result(timeout=args.timeout)
                        if res:
                            results.append(res)
                            completed += 1
                    except Exception as e:
                        logging.error('Error in worker: %s', e)
                        logging.debug(traceback.format_exc())
            else:
                for fut in as_completed(futures):
                    try:
                        res = fut.result(timeout=args.timeout)
                        if res:
                            results.append(res)
                            completed += 1
                        if completed % progress_interval == 0:
                            logging.info('Progress: %d / %d', completed, total_files)
                    except Exception as e:
                        logging.error('Error in worker: %s', e)
                        logging.debug(traceback.format_exc())

            if (
                results
                and _PANDAS_AVAILABLE
                and ('agg-csv' in outputs_map or 'agg-json' in outputs_map)
            ):
                df = pd.DataFrame(results)
                if 'agg-csv' in outputs_map:
                    combined_csv = Path(args.output_dir) / 'combined_metadata.csv'
                    if args.dry_run:
                        logging.info('DRY RUN: would write combined CSV -> %s', combined_csv)
                    else:
                        try:
                            df.to_csv(combined_csv, index=False)
                            logging.info('Combined CSV -> %s', combined_csv)
                        except Exception as e:
                            logging.error('Failed to write combined CSV: %s', e)
                if 'agg-json' in outputs_map:
                    combined_json = Path(args.output_dir) / 'combined_metadata.json'
                    if args.dry_run:
                        logging.info('DRY RUN: would write combined JSON -> %s', combined_json)
                    else:
                        try:
                            df.to_json(combined_json, orient='records', force_ascii=False, indent=2)
                            logging.info('Combined JSON -> %s', combined_json)
                        except Exception as e:
                            logging.error('Failed to write combined JSON: %s', e)
            elif results and (
                ('agg-csv' in outputs_map or 'agg-json' in outputs_map) and not _PANDAS_AVAILABLE
            ):
                logging.warning(
                    'Tip: install pandas to create a combined CSV/JSON for all batch files (pip install pandas)'
                )

        # Save mapping
        if args.anonymize and args.anonymize_map and run_mapping:
            try:
                amap_path = Path(args.anonymize_map)
                amap_path.parent.mkdir(parents=True, exist_ok=True)
                with open(amap_path, 'w', encoding='utf-8') as f:
                    json.dump(run_mapping, f, indent=2, ensure_ascii=False)
                logging.info('Saved anonymization map -> %s', amap_path)
            except Exception as e:
                logging.error('Failed to save anonymization map: %s', e)

        logging.info('Batch complete. Reports (if any) in: %s', args.output_dir)
        return

    # Single file non-batch mode
    if not args.path:
        logging.error('No path provided and not interactive. Use --batch or provide a file path.')
        sys.exit(1)
    if not Path(args.path).exists():
        logging.error('Path does not exist: %s', args.path)
        sys.exit(1)

    mapping = {}
    process_and_save(
        args.path, args, outputs_map, mapping, dry_run=args.dry_run, suppress_details=False
    )


if __name__ == '__main__':
    main()
