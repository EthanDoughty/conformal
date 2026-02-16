#!/usr/bin/env python3
# Entry point for LSP server: python3 -m lsp

import sys
from pathlib import Path

# Ensure repo root is on sys.path (needed when spawned from VS Code)
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from lsp.server import server

if __name__ == "__main__":
    server.start_io()
