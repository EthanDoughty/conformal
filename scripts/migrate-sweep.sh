#!/usr/bin/env bash
# Sweep the dogfood corpus through conformal-migrate and report gap metrics.
#
# Sample: every 10th .m file (sorted path order, corpus-v1 excluded) so runs
# are deterministic and comparable across passes. Output lands in a work dir
# (default /tmp/migrate-sweep) holding one .py per input plus summary files.
#
# Usage: scripts/migrate-sweep.sh [workdir]

set -u
BIN="$(dirname "$0")/../src/migrate/bin/Release/net8.0/conformal-migrate"
CORPUS="/root/projects/MATLAB_analysis/dogfood"
WORK="${1:-/tmp/migrate-sweep}"
JOBS=8

if [ ! -x "$BIN" ]; then
    echo "error: Release binary not found at $BIN (build first)" >&2
    exit 1
fi

rm -rf "$WORK"
mkdir -p "$WORK/py"

find "$CORPUS" -name '*.m' -not -path '*/corpus-v1/*' | sort | awk 'NR % 10 == 1' > "$WORK/sample.txt"
TOTAL=$(wc -l < "$WORK/sample.txt")
echo "Sweeping $TOTAL files with $JOBS jobs..."

export BIN WORK CORPUS
run_one() {
    f="$1"
    rel="${f#"$CORPUS"/}"
    out="$WORK/py/${rel//\//__}.py"
    err=$(timeout 20 "$BIN" "$f" --stdout 2>&1 >"$out")
    code=$?
    if [ $code -ne 0 ]; then
        echo "$rel|$code|$(echo "$err" | head -1)" >> "$WORK/failures.txt"
        rm -f "$out"
    fi
}
export -f run_one
xargs -a "$WORK/sample.txt" -P "$JOBS" -I {} bash -c 'run_one "$@"' _ {}

FAILED=$(wc -l < "$WORK/failures.txt" 2>/dev/null || echo 0)
OK=$((TOTAL - FAILED))
echo "Translated: $OK/$TOTAL (failures: $FAILED, see failures.txt)"

# Python syntax check over all outputs in one interpreter run
python3 - "$WORK" <<'EOF'
import sys, os
work = sys.argv[1]
pydir = os.path.join(work, "py")
bad = []
for name in sorted(os.listdir(pydir)):
    path = os.path.join(pydir, name)
    with open(path, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    try:
        compile(src, name, "exec")
    except SyntaxError as e:
        bad.append(f"{name}|line {e.lineno}|{e.msg}")
with open(os.path.join(work, "syntax_errors.txt"), "w") as fh:
    fh.write("\n".join(bad) + ("\n" if bad else ""))
print(f"Syntax errors: {len(bad)} files (see syntax_errors.txt)")
EOF

# Opaque-line extraction: '# MATLAB:' passthroughs with non-blank content
grep -h '# MATLAB:' "$WORK"/py/*.py 2>/dev/null \
    | sed 's/^[[:space:]]*# MATLAB:[[:space:]]*//' \
    | grep -v '^$' > "$WORK/opaque_lines.txt" || true
OPAQUE=$(wc -l < "$WORK/opaque_lines.txt")
OPAQUE_FILES=$(grep -l '# MATLAB:' "$WORK"/py/*.py 2>/dev/null | wc -l)
CLEAN=$((OK - OPAQUE_FILES))
echo "Opaque lines (non-blank): $OPAQUE across $OPAQUE_FILES files"
echo "Clean files (translated, zero opaque): $CLEAN"

# Top opaque patterns by leading token
awk '{print $1}' "$WORK/opaque_lines.txt" | sort | uniq -c | sort -rn | head -30 > "$WORK/opaque_top.txt"
echo "--- top opaque leading tokens (opaque_top.txt) ---"
cat "$WORK/opaque_top.txt"
