#!/usr/bin/env bash
set -euo pipefail

MODEL="${LMSTUDIO_MODEL:-qwen/qwen3-14b}"
PORT="${LMSTUDIO_PORT:-1234}"
CTX="${LMSTUDIO_CTX:-8192}"
GPU="${LMSTUDIO_GPU:-max}"
TTL="${LMSTUDIO_TTL:-900}"   # 15 minutes idle auto-unload

# Load model (idempotent-ish: if already loaded, LM Studio will just keep it loaded)
lms load "$MODEL" --gpu "$GPU" --context-length "$CTX" --ttl "$TTL" -y

# Start server
lms server start -p "$PORT"

echo "LM Studio ready:"
echo "  BASE_URL=http://localhost:${PORT}/v1"
echo "  MODEL=${MODEL}"