#!/usr/bin/env python3
# Entry point for LSP server: python3 -m lsp

import sys
from pathlib import Path

try:
    from lsp.server import server
except ImportError:
    # Fallback for running from repo checkout without pip install
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from lsp.server import server

if __name__ == "__main__":
    server.start_io()
