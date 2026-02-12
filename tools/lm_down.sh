#!/usr/bin/env bash
set -euo pipefail
lms server stop || true
lms unload --all || true