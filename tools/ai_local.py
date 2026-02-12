from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import requests

BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
MODEL = os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-14b")

DEFAULT_SYSTEM = (
    "You are a professional code collaborator for this repo. "
    "Do not reveal chain-of-thought. Output only final answers. "
    "Be concise, architecture-aware, and follow AGENTS.md + TASK.md."
)

THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def chat(system: str, user: str) -> str:
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def build_system(role: str) -> str:
    agents = read_text("AGENTS.md") if Path("AGENTS.md").exists() else ""
    task = read_text("TASK.md") if Path("TASK.md").exists() else ""
    return (
        DEFAULT_SYSTEM
        + f"\n\nROLE={role}\n\nAGENTS.md:\n{agents}\n\nTASK.md:\n{task}\n"
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Local LM Studio agent runner")
    p.add_argument("role", nargs="?", default="mentor", help="role name (mentor/tester/implementer/etc.)")
    p.add_argument("--file", action="append", default=[], help="additional file(s) to include in user prompt")
    args = p.parse_args(argv)

    # stdin is the actual instruction/question
    user_prompt = sys.stdin.read().strip()

    # optionally append file contents (diffs, logs, snippets)
    if args.file:
        chunks: list[str] = []
        for fp in args.file:
            path = Path(fp)
            if not path.exists():
                chunks.append(f"[Missing file: {fp}]")
                continue
            chunks.append(f"=== FILE: {fp} ===\n{path.read_text(encoding='utf-8')}")
        if user_prompt:
            user_prompt += "\n\n"
        user_prompt += "\n\n".join(chunks)

    if not user_prompt:
        user_prompt = "Describe what you can do for ROLE and how to use you in this repo."

    system = build_system(args.role)
    out = chat(system, user_prompt)
    print(strip_think(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))