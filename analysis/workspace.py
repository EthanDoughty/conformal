# Ethan Doughty
# workspace.py
"""Workspace scanning for multi-file MATLAB projects."""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ExternalSignature:
    """Signature of a function defined in an external .m file."""
    name: str           # Internal function name (from function declaration)
    param_count: int    # Number of input parameters
    return_count: int   # Number of output variables (0 = procedure)
    source_path: str    # Absolute path to the .m file (for diagnostics)


# Regex to extract first function signature from MATLAB source
# Handles 3 forms:
#   function [a, b] = name(params)    # multi-return
#   function result = name(params)    # single-return
#   function name(params)             # procedure
_FUNC_SIG_RE = re.compile(
    r'^\s*function\s+'
    r'(?:'
        r'\[([^\]]*)\]\s*=\s*(\w+)'   # multi-return: [a, b] = name
        r'|(\w+)\s*=\s*(\w+)'          # single-return: result = name
        r'|(\w+)'                        # procedure: name
    r')\s*\(([^)]*)\)',
    re.MULTILINE
)


def extract_function_signature(source: str) -> Optional[Tuple[str, int, int]]:
    """Extract the first function signature from MATLAB source code.

    Args:
        source: MATLAB source code as string

    Returns:
        (function_name, param_count, return_count) or None if no function found
    """
    match = _FUNC_SIG_RE.search(source)
    if not match:
        return None

    groups = match.groups()
    # groups: (multi_rets, multi_name, single_ret, single_name, proc_name, params)
    multi_rets, multi_name, single_ret, single_name, proc_name, params = groups

    if multi_rets is not None:
        # Multi-return form: [a, b] = name(...)
        func_name = multi_name
        return_vars = [v.strip() for v in multi_rets.split(',') if v.strip()]
        return_count = len(return_vars)
    elif single_ret is not None:
        # Single-return form: result = name(...)
        func_name = single_name
        return_count = 1
    else:
        # Procedure form: name(...)
        func_name = proc_name
        return_count = 0

    # Extract param count
    param_list = [p.strip() for p in params.split(',') if p.strip()]
    param_count = len(param_list)

    return (func_name, param_count, return_count)


def scan_workspace(directory: Path, exclude: str = None) -> Dict[str, ExternalSignature]:
    """Scan a directory for .m files and extract function signatures.

    Args:
        directory: Directory to scan
        exclude: Filename to exclude (typically the file being analyzed)

    Returns:
        Dict mapping filename stems to ExternalSignature objects
    """
    result = {}

    if not directory.is_dir():
        return result

    for file_path in directory.glob('*.m'):
        # Skip the excluded file (the one being analyzed)
        if exclude and file_path.name == exclude:
            continue

        try:
            source = file_path.read_text(encoding='utf-8')
            sig_tuple = extract_function_signature(source)
            if sig_tuple is None:
                # Script file or no function found
                continue

            func_name, param_count, return_count = sig_tuple
            # Key by filename stem (MATLAB dispatch semantics)
            key = file_path.stem
            result[key] = ExternalSignature(
                name=func_name,
                param_count=param_count,
                return_count=return_count,
                source_path=str(file_path.resolve())
            )
        except (OSError, UnicodeDecodeError):
            # File read failed; skip silently (conservative)
            continue

    return result
