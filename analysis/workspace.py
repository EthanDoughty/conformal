# Ethan Doughty
# workspace.py
"""Workspace scanning for multi-file MATLAB projects."""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from analysis.context import FunctionSignature
from ir import FunctionDef


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
    r')\s*(?:\(([^)]*)\))?',
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
        return_vars = [v.strip() for v in re.split(r'[,\s]+', multi_rets) if v.strip()]
        return_count = len(return_vars)
    elif single_ret is not None:
        # Single-return form: result = name(...)
        func_name = single_name
        return_count = 1
    else:
        # Procedure form: name(...)
        func_name = proc_name
        return_count = 0

    # Extract param count (params is None when no parentheses present)
    if params is None:
        param_count = 0
    else:
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
            source = file_path.read_text(encoding='utf-8', errors='replace')
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


# Cache of parsed external files (keyed by absolute source_path)
_parsed_cache: Dict[str, object] = {}


def clear_parse_cache() -> None:
    """Clear the parsed external file cache. Useful for LSP cache invalidation."""
    _parsed_cache.clear()


def load_external_function(sig: ExternalSignature) -> Optional[Tuple[FunctionSignature, Dict[str, FunctionSignature]]]:
    """Load and parse an external .m file, extracting its primary function and subfunctions.

    Args:
        sig: ExternalSignature from workspace scanning

    Returns:
        (primary_FunctionSignature, subfunctions_dict) or None on any error
    """
    from frontend.pipeline import parse_syntax, lower_to_ir

    source_path = sig.source_path
    if source_path in _parsed_cache:
        ir_prog = _parsed_cache[source_path]
    else:
        try:
            source = Path(source_path).read_text(encoding='utf-8', errors='replace')
            syntax_ast = parse_syntax(source)
            ir_prog = lower_to_ir(syntax_ast)
            _parsed_cache[source_path] = ir_prog
        except Exception:
            return None

    primary = None
    subfunctions: Dict[str, FunctionSignature] = {}
    for item in ir_prog.body:
        if isinstance(item, FunctionDef):
            func_sig = FunctionSignature(
                name=item.name, params=item.params,
                output_vars=item.output_vars, body=item.body
            )
            if primary is None:
                primary = func_sig
            else:
                subfunctions[item.name] = func_sig

    if primary is None:
        return None

    return (primary, subfunctions)
