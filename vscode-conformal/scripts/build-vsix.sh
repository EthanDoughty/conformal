#!/usr/bin/env bash
# Build pipeline: Fable compile -> esbuild bundle -> smoke test -> VSIX package.
# Fails fast on any step. Run from repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Fable compile ==="
cd "$REPO_ROOT/src"
TMPDIR="${TMPDIR:-/tmp}" dotnet fable ../vscode-conformal/fable/ConformalFable.fsproj --outDir ../vscode-conformal/src/fable-out

echo "=== esbuild bundle ==="
cd "$REPO_ROOT/vscode-conformal"
node esbuild.mjs

echo "=== Smoke test ==="
node scripts/smoke-test.cjs

echo "=== VSIX package ==="
npx @vscode/vsce package --allow-missing-repository

echo "=== Done ==="
ls -lh "$REPO_ROOT"/vscode-conformal/*.vsix
