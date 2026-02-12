"""Local LLM agent runner via Ollama.

Usage:
    python3 tools/ai_local.py [--role ROLE] [--file FILE ...] [--no-context] [--raw] PROMPT ...
    python3 tools/ai_local.py --chat [--role ROLE] [--file FILE ...] [--no-context]
    echo "question" | python3 tools/ai_local.py [--role ROLE] [--file FILE ...]

Sends a prompt to a local Qwen 2.5 Coder 7B (or other model) via Ollama's
OpenAI-compatible API.  Auto-starts the server when it isn't running.

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

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
CTX_LIMIT = int(os.getenv("OLLAMA_CTX", "32768"))

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


def _server_healthy() -> bool:
    """Return True if Ollama API responds."""
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=3)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def _ensure_server(timeout: float = 15.0) -> None:
    """Check health; auto-start Ollama server if needed."""
    if _server_healthy():
        return

    _eprint("Server not running — starting Ollama...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
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
    usage = data.get("usage", {})
    return data["choices"][0]["message"]["content"], usage


def _format_usage(usage: dict) -> str:
    """Format token usage as a compact status string."""
    if not usage:
        return ""
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", 0)
    pct = total / CTX_LIMIT * 100 if CTX_LIMIT else 0
    return f"[tokens: {prompt} in + {completion} out = {total} total — {pct:.0f}% of {CTX_LIMIT} ctx]"


def build_system(role: str, *, include_context: bool = True) -> str:
    parts = [DEFAULT_SYSTEM, f"\nROLE={role}"]

    if include_context:
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

    args = p.parse_args(argv)

    # --- Ensure server is running -----------------------------------------
    _ensure_server()

    system = build_system(args.role, include_context=not args.no_context)

    # --- Build file attachment (shared by both modes) ---------------------
    file_attachment = ""
    if args.file:
        chunks: list[str] = []
        for fp in args.file:
            path = Path(fp)
            if not path.exists():
                chunks.append(f"[Missing file: {fp}]")
                continue
            chunks.append(f"=== FILE: {fp} ===\n{path.read_text(encoding='utf-8')}")
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
