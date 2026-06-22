#!/usr/bin/env bash
# check-doc-numbers.sh: flag drift between literal numbers in the docs and the
# values derived from the source tree.
#
# Test counts, version strings, and builtin totals outlive their accuracy as the
# code moves under them, and a reader trusts a stale number as much as a fresh
# one. The version and the migrate builtin total come straight from source and
# are checked exactly. The test total is not cheaply derivable, since workspace
# fixtures span several files, so the docs are checked against each other and
# --full runs the suite for the authoritative count. The two compliance docs are
# pinned to a release on purpose, so their older numbers are reported, not failed.
#
# Usage: scripts/check-doc-numbers.sh [--full] [--list] [extra-doc ...]
#   --full   run the test suite for the authoritative test count
#   --list   list every literal number in doc prose, then exit
# Exit status is nonzero when a tracked number disagrees.

set -u
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "not in a git repo" >&2; exit 2; }
cd "$ROOT" || exit 2

full=0; list=0; extra=()
for a in "$@"; do
  case "$a" in
    --full) full=1 ;;
    --list) list=1 ;;
    *) extra+=("$a") ;;
  esac
done

# Reader-facing docs. CHANGELOG is excluded on purpose: it records every past
# version and count by design.
DOCS=(
  README.md
  vscode-conformal/README.md
  docs/analysis.md
  docs/tests.md
  docs/TOR.md
  docs/false-negative-policy.md
  site/index.html
  site/llms.txt
)
[ "${#extra[@]}" -gt 0 ] && DOCS+=("${extra[@]}")

# Compliance docs pinned to a release. Their version and test counts are scoped
# to that release, so a mismatch here is information, not a failure.
is_scoped() { case "$1" in docs/TOR.md|docs/false-negative-policy.md) return 0 ;; *) return 1 ;; esac; }

# Strip a doc down to prose for matching: drop fenced code in markdown, drop
# tags and script/style in HTML. Line numbers are preserved as blanks.
prose() {
  case "$1" in
    *.html|*.htm)
      awk 'BEGIN{s=0} /<style|<script/{s=1} {if(s){print ""}else{l=$0;gsub(/<[^>]*>/,"",l);print l}} /<\/style>|<\/script>/{s=0}' "$1" ;;
    *)
      awk 'BEGIN{f=0} /^[[:space:]]*```/{f=!f;print "";next} {if(f)print "";else print}' "$1" ;;
  esac
}

red(){ printf '\033[31m%s\033[0m' "$1"; }
grn(){ printf '\033[32m%s\033[0m' "$1"; }
ylw(){ printf '\033[33m%s\033[0m' "$1"; }

fails=0

# --- Derived ground truth -------------------------------------------------
VERSION=$(grep -m1 '"version"' vscode-conformal/package.json | sed -E 's/.*:[[:space:]]*"([^"]+)".*/\1/')
CLI_VERS=$(grep -oE '"3\.[0-9]+\.[0-9]+"' src/analyzer/Cli.fs 2>/dev/null | tr -d '"' | sort -u)
MIGRATE_BUILTINS=$(grep -cE '^[[:space:]]*"[a-zA-Z0-9_]+"[[:space:]]*,[[:space:]]*\{[[:space:]]*pythonFunc' src/migrate/BuiltinMap.fs)

# --list: dump every three-or-more-digit number in doc prose and exit.
if [ "$list" -eq 1 ]; then
  echo "# Literal numbers in doc prose (>= 3 digits)"
  for d in "${DOCS[@]}"; do
    [ -f "$d" ] || continue
    hits=$(prose "$d" | grep -vE 'https?://' | grep -noE '[0-9][0-9,]{2,}\+?')
    [ -n "$hits" ] && { echo "## $d"; echo "$hits" | sed 's/^/  /'; }
  done
  exit 0
fi

echo "# Doc number check"
echo "  derived version          $(grn "$VERSION")  (vscode-conformal/package.json)"
if [ "$(echo "$CLI_VERS" | wc -l)" -ne 1 ] || [ "$CLI_VERS" != "$VERSION" ]; then
  echo "  $(red FAIL) src/analyzer/Cli.fs version literal(s) [$CLI_VERS] disagree with package.json [$VERSION]"
  fails=$((fails+1))
else
  echo "  source version literals  $(grn "agree") (Cli.fs matches package.json)"
fi
echo "  derived migrate builtins $(grn "$MIGRATE_BUILTINS")  (src/migrate/BuiltinMap.fs)"
echo ""

# --- Version drift in docs ------------------------------------------------
echo "## Version ($VERSION expected)"
for d in "${DOCS[@]}"; do
  [ -f "$d" ] || continue
  while IFS=: read -r ln val; do
    [ -z "$val" ] && continue
    norm=${val#v}
    [ "$norm" = "$VERSION" ] && continue
    if is_scoped "$d"; then
      printf '  %s %s:%s pinned to %s\n' "$(ylw INFO)" "$d" "$ln" "$norm"
    else
      printf '  %s %s:%s says %s\n' "$(red FAIL)" "$d" "$ln" "$norm"
      fails=$((fails+1))
    fi
  done < <(prose "$d" | grep -noE '\bv?3\.[0-9]+\.[0-9]+\b')
done
echo ""

# --- Test-count consistency ----------------------------------------------
# Capture the count from each phrasing the docs actually use.
test_counts() {
  prose "$1" | grep -oE 'tests-[0-9]{3}|[0-9]{3}[ -]tests?\b|[0-9]{3} self-checking|[0-9]{3} integration tests?|[0-9]{3} test cases|contains [0-9]{3}' \
    | grep -oE '[0-9]{3}' | sort -u
}
echo "## Test count"
declare -A seen_vals
canon_locs=""
for d in "${DOCS[@]}"; do
  [ -f "$d" ] || continue
  vals=$(test_counts "$d")
  [ -z "$vals" ] && continue
  for v in $vals; do
    if is_scoped "$d"; then
      printf '  %s %s states %s (pinned to a release)\n' "$(ylw INFO)" "$d" "$v"
    else
      printf '  %s %s states %s\n' "  -" "$d" "$v"
      seen_vals["$v"]=1
      canon_locs="$canon_locs $d:$v"
    fi
  done
done
distinct=${#seen_vals[@]}
if [ "$distinct" -gt 1 ]; then
  echo "  $(red FAIL) docs disagree on the test count: ${!seen_vals[*]}"
  fails=$((fails+1))
elif [ "$distinct" -eq 1 ]; then
  agreed=${!seen_vals[*]}
  echo "  $(grn "agree") on $agreed"
  if [ "$full" -eq 1 ]; then
    echo "  running the suite for the authoritative count ..."
    sum=$(dotnet run --no-build --project src/analyzer/ConformalAnalyzer.fsproj -c Release -- --tests --quiet 2>/dev/null | grep -oE 'Summary: [0-9]+/[0-9]+' | grep -oE '[0-9]+' | head -1)
    if [ -z "$sum" ]; then
      echo "  $(ylw INFO) could not run the suite (build Release first); skipped authoritative check"
    elif [ "$sum" != "$agreed" ]; then
      echo "  $(red FAIL) docs say $agreed but the suite reports $sum"
      fails=$((fails+1))
    else
      echo "  $(grn "confirmed") against the suite ($sum)"
    fi
  fi
fi
echo ""

# --- Migrate builtin claims (informational) -------------------------------
echo "## Migrate builtins ($MIGRATE_BUILTINS in source)"
found=0
for d in "${DOCS[@]}"; do
  [ -f "$d" ] || continue
  while IFS=: read -r ln line; do
    echo "$line" | grep -qiE 'migrate|transpiler|numpy|np\.dot' || continue
    num=$(echo "$line" | grep -oE '[0-9]{2,}\+? (MATLAB )?builtins' | grep -oE '[0-9]+' | head -1)
    [ -z "$num" ] && continue
    found=1
    if [ "$num" = "$MIGRATE_BUILTINS" ]; then
      printf '  %s %s:%s states %s\n' "$(grn OK)" "$d" "$ln" "$num"
    else
      printf '  %s %s:%s states %s (source has %s)\n' "$(red FAIL)" "$d" "$ln" "$num" "$MIGRATE_BUILTINS"
      fails=$((fails+1))
    fi
  done < <(prose "$d" | grep -niE '[0-9]{2,}\+? (MATLAB )?builtins')
done
[ "$found" -eq 0 ] && echo "  (no numeric migrate-builtin claims in the scanned docs)"
echo ""

# --- Verdict --------------------------------------------------------------
if [ "$fails" -gt 0 ]; then
  echo "$(red "FAIL") $fails drifted number(s). Update the docs or the source."
  exit 1
fi
echo "$(grn "OK") doc numbers agree with the source."
