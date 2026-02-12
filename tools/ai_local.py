"""Local LLM agent runner via Ollama.

Usage:
    python3 tools/ai_local.py [--role ROLE] [--file FILE ...] [--no-context] [--raw] PROMPT ...
    python3 tools/ai_local.py --chat [--role ROLE] [--file FILE ...] [--no-context]
    python3 tools/ai_local.py --summarize --file FILE [--file FILE ...]
    echo "question" | python3 tools/ai_local.py [--role ROLE] [--file FILE ...]

Sends a prompt to a local Qwen 2.5 Coder 14B (or other model) via Ollama's
native API.  Auto-starts the server when it isn't running.

Use --chat for an interactive REPL that maintains conversation history.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")
CTX_LIMIT = int(os.getenv("OLLAMA_CTX", "12288"))

DEFAULT_SYSTEM = (
    "You are a professional code collaborator for this repo. "
    "Do not reveal chain-of-thought. Output only final answers. "
    "Be concise, architecture-aware, and follow project conventions."
)

THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)

# Project root: two levels up from tools/ai_local.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eprint(*args, **kwargs):
    """Print to stderr (keeps stdout clean for piping)."""
    print(*args, file=sys.stderr, **kwargs)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def _read_file_spec(spec: str) -> tuple[str, str]:
    """Parse 'path:start-end' → (label, content). Plain 'path' returns full file."""
    if ":" in spec and spec.rsplit(":", 1)[1].replace("-", "").isdigit():
        path_str, range_str = spec.rsplit(":", 1)
        parts = range_str.split("-", 1)
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 else start
        path = Path(path_str)
        if not path.exists():
            return spec, f"[Missing file: {path_str}]"
        lines = path.read_text(encoding="utf-8").splitlines()
        selected = lines[start - 1 : end]  # 1-indexed
        return f"{path_str}:{start}-{end}", "\n".join(selected)
    path = Path(spec)
    if not path.exists():
        return spec, f"[Missing file: {spec}]"
    return spec, path.read_text(encoding="utf-8")


def _compact(text: str) -> str:
    """Strip comment-only lines and blank lines."""
    return "\n".join(
        line for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "%"))
    )


# --- Summary caching ----------------------------------------------------------

CACHE_DIR = PROJECT_ROOT / ".cache" / "summaries"

SUMMARIZE_PROMPT = (
    "Summarize this file in 3-5 bullet points. "
    "Focus on purpose, key functions, and public API.\n\n"
)


def _cache_path(file_path: Path) -> Path:
    safe = str(file_path.resolve().relative_to(PROJECT_ROOT)).replace("/", "__")
    return CACHE_DIR / f"{safe}.md"


def _get_cached_summary(file_path: Path) -> str | None:
    cp = _cache_path(file_path)
    if not cp.exists():
        return None
    text = cp.read_text(encoding="utf-8")
    first_line, _, body = text.partition("\n")
    try:
        cached_mtime = float(first_line.split(":", 1)[1].strip().rstrip("-->").strip())
    except (IndexError, ValueError):
        return None
    if file_path.stat().st_mtime != cached_mtime:
        return None
    return body.strip()


def _save_cached_summary(file_path: Path, summary: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mtime = file_path.stat().st_mtime
    _cache_path(file_path).write_text(
        f"<!-- mtime:{mtime} -->\n{summary}", encoding="utf-8"
    )


def _server_healthy() -> bool:
    """Return True if Ollama API responds."""
    try:
        r = requests.get(f"{BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def _ensure_server(timeout: float = 15.0) -> None:
    """Check health; auto-start Ollama server if needed."""
    if _server_healthy():
        return

    _eprint("Server not running — starting Ollama...")
    env = os.environ.copy()
    env.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    env.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")
    env.setdefault("OLLAMA_NUM_CTX", "32000")
    env.setdefault("OLLAMA_NUM_GPU", "999")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(1)
        if _server_healthy():
            _eprint("Server ready.")
            return

    raise SystemExit(
        f"ERROR: Ollama server did not become healthy within {timeout}s.\n"
        "  Start manually: ollama serve"
    )


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def chat(messages: list[dict[str, str]]) -> tuple[str, dict]:
    """Send messages and return (content, usage_dict).

    usage_dict has keys: prompt_tokens, completion_tokens, total_tokens.
    Falls back to empty dict if the server omits usage info.
    """
    url = f"{BASE_URL}/api/chat"
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": CTX_LIMIT,
        },
    }
    r = requests.post(url, json=payload, timeout=300)
    if r.status_code == 400:
        body = (
            r.json()
            if r.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        err = body.get("error", r.text)
        if "context" in str(err).lower():
            raise SystemExit(
                f"ERROR: Prompt exceeds model context window.\n"
                f"  Server said: {err}\n"
                f"  Try: reduce --file contents, or use a model with larger context"
            )
        raise SystemExit(f"ERROR: 400 Bad Request — {err}")
    r.raise_for_status()
    data = r.json()
    usage = {
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
    }
    return data["message"]["content"], usage


def _format_usage(usage: dict) -> str:
    """Format token usage as a compact status string."""
    if not usage:
        return ""
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", 0)
    pct = total / CTX_LIMIT * 100 if CTX_LIMIT else 0
    return f"[tokens: {prompt} in + {completion} out = {total} total — {pct:.0f}% of {CTX_LIMIT} ctx]"


LITE_CONTEXT = """\
PROJECT: Mini-MATLAB static shape analyzer (matrix dimension checking before runtime).
PIPELINE: frontend/matlab_parser.py → frontend/lower_ir.py → analysis/analysis_ir.py (authoritative).
IR: ir/ir.py (dataclass AST). Shapes: runtime/shapes.py (scalar | matrix[r x c] | unknown).
Dims: int (concrete) | str (symbolic) | None (unknown). Key: join_dim, dims_definitely_conflict, add_dim.
Env: runtime/env.py (var→shape map, join_env merges branches).
Diagnostics: analysis/diagnostics.py (W_* warning codes).
Tests: tests/test*.m with inline % EXPECT: assertions. Runner: run_all_tests.py.
INVARIANT: IR analyzer is source of truth. Legacy analyzer is for comparison only.
"""


def build_system(role: str, *, include_context: bool = True, lite: bool = False) -> str:
    parts = [DEFAULT_SYSTEM, f"\nROLE={role}"]

    if lite:
        parts.append(f"\n{LITE_CONTEXT}")
    elif include_context:
        for name in ("CLAUDE.md", "AGENTS.md", "TASK.md"):
            p = PROJECT_ROOT / name
            if p.exists():
                parts.append(f"\n{name}:\n{read_text(p)}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Local Ollama agent runner",
        usage="%(prog)s [--role ROLE] [--file FILE ...] [--no-context] [--raw] [PROMPT ...]",
    )
    p.add_argument(
        "prompt", nargs="*", default=[],
        help="prompt text (joined with spaces); reads stdin if omitted and stdin is piped",
    )
    p.add_argument(
        "--role", "-r", default="mentor",
        help="role name: mentor, implementer, tester, etc. (default: mentor)",
    )
    p.add_argument(
        "--file", action="append", default=[],
        help="attach file contents to the prompt (repeatable)",
    )
    p.add_argument(
        "--no-context", action="store_true",
        help="skip injecting CLAUDE.md / AGENTS.md / TASK.md into system prompt",
    )
    p.add_argument(
        "--raw", action="store_true",
        help="keep <think> tags in output (skip stripping)",
    )
    p.add_argument(
        "--chat", action="store_true",
        help="interactive REPL with conversation history (type /quit to exit)",
    )
    p.add_argument(
        "--compact", action="store_true",
        help="strip comment lines and blank lines from file attachments",
    )
    p.add_argument(
        "--summarize", action="store_true",
        help="generate and cache file summaries (use with --file)",
    )
    p.add_argument(
        "--lite-context", action="store_true",
        help="use compact ~400-token project summary instead of full docs",
    )

    args = p.parse_args(argv)

    # --- Summarize mode (no server needed for cache hits) -----------------
    if args.summarize:
        if not args.file:
            _eprint("--summarize requires at least one --file argument.")
            return 1
        return _summarize_files(args.file, role=args.role, raw=args.raw)

    # --- Ensure server is running -----------------------------------------
    _ensure_server()

    system = build_system(
        args.role,
        include_context=not args.no_context,
        lite=args.lite_context,
    )

    # --- Build file attachment (shared by both modes) ---------------------
    file_attachment = ""
    if args.file:
        chunks: list[str] = []
        for fp in args.file:
            label, content = _read_file_spec(fp)
            if content.startswith("[Missing file:"):
                chunks.append(content)
                continue
            if args.compact:
                content = _compact(content)
            chunks.append(f"=== FILE: {label} ===\n{content}")
        file_attachment = "\n\n".join(chunks)

    # --- Interactive chat mode --------------------------------------------
    if args.chat:
        return _chat_loop(system, file_attachment, raw=args.raw)

    # --- One-shot mode ----------------------------------------------------
    # Priority: positional args > piped stdin > fallback
    if args.prompt:
        user_prompt = " ".join(args.prompt)
    elif not sys.stdin.isatty():
        user_prompt = sys.stdin.read().strip()
    else:
        p.print_help(sys.stderr)
        return 1

    if file_attachment:
        user_prompt = f"{user_prompt}\n\n{file_attachment}" if user_prompt else file_attachment

    if not user_prompt:
        _eprint("No prompt provided.")
        return 1

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    _eprint(f"Asking {MODEL} (role={args.role})...")
    out, usage = chat(messages)

    if args.raw:
        print(out)
    else:
        print(strip_think(out))
    usage_str = _format_usage(usage)
    if usage_str:
        _eprint(usage_str)
    return 0


def _summarize_files(file_specs: list[str], *, role: str, raw: bool) -> int:
    """Generate or return cached summaries for each file spec."""
    need_llm: list[tuple[Path, str]] = []  # (resolved_path, content)
    results: list[tuple[str, str, bool]] = []  # (label, summary, was_cached)

    for spec in file_specs:
        label, content = _read_file_spec(spec)
        if content.startswith("[Missing file:"):
            results.append((label, content, False))
            continue
        # Resolve to actual file path (strip line range for caching)
        path_str = spec.rsplit(":", 1)[0] if ":" in spec and spec.rsplit(":", 1)[1].replace("-", "").isdigit() else spec
        file_path = Path(path_str).resolve()
        cached = _get_cached_summary(file_path)
        if cached is not None:
            results.append((label, cached, True))
        else:
            need_llm.append((file_path, content))
            results.append((label, "", False))  # placeholder

    if need_llm:
        _ensure_server()
        system = build_system(role, lite=True)
        for file_path, content in need_llm:
            _eprint(f"Summarizing {file_path.name}...")
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{SUMMARIZE_PROMPT}{content}"},
            ]
            reply, usage = chat(messages)
            summary = reply if raw else strip_think(reply)
            _save_cached_summary(file_path, summary)
            # Fill in the placeholder
            for i, (lbl, _, was_cached) in enumerate(results):
                if not was_cached and results[i][1] == "":
                    rel = file_path.relative_to(Path.cwd()) if file_path.is_relative_to(Path.cwd()) else file_path
                    if str(rel) in lbl or file_path.name in lbl:
                        results[i] = (lbl, summary, False)
                        break
            usage_str = _format_usage(usage)
            if usage_str:
                _eprint(usage_str)

    for label, summary, was_cached in results:
        tag = "(cached)" if was_cached else "(generated)"
        print(f"=== {label} {tag} ===\n{summary}\n")

    return 0


def _chat_loop(system: str, file_attachment: str, *, raw: bool) -> int:
    """Interactive REPL with full conversation history."""
    try:
        import readline  # noqa: F401 — enables line editing in input()
    except ImportError:
        pass

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    last_usage: dict = {}

    # If files were attached, prepend them as context in the first system turn
    if file_attachment:
        messages[0]["content"] += f"\n\nAttached files:\n{file_attachment}"

    _eprint(f"Chat with {MODEL} — type /quit or Ctrl-D to exit, /reset to clear history")
    _eprint(f"Context window: {CTX_LIMIT} tokens")
    _eprint("")

    try:
        while True:
            try:
                line = input("you> ")
            except EOFError:
                _eprint("\nBye.")
                return 0

            line = line.strip()
            if not line:
                continue
            if line in ("/quit", "/exit", "/q"):
                _eprint("Bye.")
                return 0
            if line == "/reset":
                messages[:] = [messages[0]]  # keep system prompt
                last_usage = {}
                _eprint("History cleared.")
                continue
            if line == "/history":
                _eprint(f"{(len(messages) - 1) // 2} exchanges in history")
                continue
            if line == "/usage":
                if last_usage:
                    _eprint(_format_usage(last_usage))
                else:
                    _eprint("No usage data yet — send a message first.")
                continue
            if line == "/help":
                _eprint("Commands: /quit /reset /history /usage /help")
                continue

            messages.append({"role": "user", "content": line})
            _eprint("thinking...")

            try:
                reply, last_usage = chat(messages)
            except (requests.ConnectionError, requests.Timeout) as exc:
                _eprint(f"Connection error: {exc}")
                messages.pop()  # remove the failed user message
                continue

            if raw:
                clean = reply
            else:
                clean = strip_think(reply)

            messages.append({"role": "assistant", "content": clean})
            usage_str = _format_usage(last_usage)
            print(f"\n{clean}\n")
            if usage_str:
                _eprint(usage_str)

    except KeyboardInterrupt:
        _eprint("\nBye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
