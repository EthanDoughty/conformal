"""Bridge to the F# conformal-parse binary."""
import os
import subprocess
import tempfile

from ir.ir import Program
from frontend.ir_json import ir_from_json

# Default paths (can be overridden via CONFORMAL_PARSE_BIN env var).
_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_DIR, '..', 'fsharp', 'bin', 'Debug', 'net8.0')
_DEFAULT_BIN = os.path.join(_BUILD_DIR, 'conformal-parse')
_DEFAULT_DLL = os.path.join(_BUILD_DIR, 'conformal-parse.dll')
_FSPROJ = os.path.join(_DIR, '..', 'fsharp', 'ConformalParse.fsproj')


def _run_fsharp(filepath: str) -> str:
    """Invoke the F# binary on *filepath* and return its JSON stdout.

    Execution strategy (fastest to slowest):
      1. ``dotnet exec <dll>``   -- if the .dll artifact exists (~0.2s/call)
      2. Prebuilt native binary  -- if CONFORMAL_PARSE_BIN points to an AOT binary
      3. ``dotnet run --project``-- full MSBuild invocation (~1.5s/call, always works)

    Args:
        filepath: Absolute path to an existing .m file.

    Returns:
        JSON string written to stdout by the F# binary.

    Raises:
        SyntaxError: if the F# binary exits non-zero (parse/lex error).
        RuntimeError: if neither strategy is available.
    """
    bin_path = os.environ.get('CONFORMAL_PARSE_BIN', _DEFAULT_BIN)
    dll_path = os.path.join(os.path.dirname(bin_path), 'conformal-parse.dll')
    if not os.path.isfile(dll_path):
        dll_path = _DEFAULT_DLL

    # Strategy 1: dotnet exec <dll> (fast — no MSBuild, just JIT startup)
    if os.path.isfile(dll_path):
        result = subprocess.run(
            ['dotnet', 'exec', dll_path, filepath],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout
        # Parse/lex error from the F# code
        if result.returncode in (2, 3):
            raise SyntaxError(
                f"F# parser error (exit {result.returncode}): {result.stderr.strip()}"
            )
        # Non-zero for other reason — fall through to try native binary or dotnet run

    # Strategy 2: native AOT binary (CONFORMAL_PARSE_BIN override)
    if os.path.isfile(bin_path) and bin_path != _DEFAULT_BIN:
        result = subprocess.run(
            [bin_path, filepath],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout
        raise SyntaxError(
            f"F# parser error (exit {result.returncode}): {result.stderr.strip()}"
        )

    # Strategy 3: dotnet run (slow but always works when SDK is installed)
    if os.path.isfile(_FSPROJ):
        result = subprocess.run(
            ['dotnet', 'run', '--project', _FSPROJ, '--', filepath],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return result.stdout
        raise SyntaxError(
            f"F# parser error (dotnet run, exit {result.returncode}): {result.stderr.strip()}"
        )

    raise RuntimeError(
        "F# frontend not available: no .dll at '{dll_path}', no binary at "
        f"'{bin_path}', and no fsproj at '{_FSPROJ}'. "
        "Build with: dotnet build fsharp/ConformalParse.fsproj"
    )


def parse_matlab_fsharp(source: str, filepath: str = None) -> Program:
    """Parse MATLAB source using the F# binary and return an IR Program.

    Drop-in replacement for ``frontend.matlab_parser.parse_matlab`` when
    the ``CONFORMAL_PARSER=fsharp`` environment variable is set.

    Args:
        source: MATLAB source code string.
        filepath: Optional path to the .m file on disk.  When supplied and
            the file exists, it is passed directly to the F# binary (avoids
            a round-trip through a temp file).  When absent or the file does
            not exist, the source is written to a temporary file.

    Returns:
        Program dataclass instance reconstructed from the F# JSON output.

    Raises:
        SyntaxError: if the F# parser reports a parse or lex error.
        RuntimeError: if the F# binary cannot be located or executed.
    """
    # Prefer passing an existing file path directly to avoid encoding issues
    if filepath and os.path.isfile(filepath):
        json_str = _run_fsharp(os.path.abspath(filepath))
        return ir_from_json(json_str)

    # Write source to a temporary file and pass that to the binary
    with tempfile.NamedTemporaryFile(suffix='.m', mode='w', encoding='utf-8',
                                     delete=False) as tmp:
        tmp.write(source)
        tmp_path = tmp.name

    try:
        json_str = _run_fsharp(tmp_path)
        return ir_from_json(json_str)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
