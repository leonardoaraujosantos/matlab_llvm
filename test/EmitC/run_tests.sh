#!/usr/bin/env bash
# Golden tests for the emit-c / emit-cpp output shape.
#
# For each .m file in this directory, runs `matlabc -emit-c -no-line`
# (and `-emit-cpp -no-line`) and diffs the generated source against
# the matching .c.expected / .cpp.expected golden. The `-no-line`
# mode is used so `#line` directives don't pin the tests to absolute
# paths that differ between build machines.
#
# Usage: run_tests.sh <path-to-matlabc>
# Env:   UPDATE=1   regenerate all .c.expected / .cpp.expected
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

UPDATE="${UPDATE:-0}"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"
pass=0; fail=0; miss=0

# Run matlabc with the given mode flag(s) and compare / refresh the golden.
# Accepts a space-separated list of extra flags as $2.
check_one() {
  local m="$1" flags="$2" ext="$3"
  local base exp tmp
  base="$(basename "${m%.m}")"
  exp="${m%.m}.${ext}"
  tmp="$(mktemp -t mlc.XXXXXX).${ext}"
  # shellcheck disable=SC2086
  if ! "$MATLABC" $flags -no-line "$m" > "$tmp" 2>/dev/null; then
    echo "FAIL $base [$ext]: matlabc $flags failed"
    fail=$((fail+1))
    rm -f "$tmp"
    return
  fi
  if [[ "$UPDATE" == "1" ]]; then
    mv "$tmp" "$exp"
    echo "UPDATE $base [$ext]"
    pass=$((pass+1))
    return
  fi
  if [[ ! -e "$exp" ]]; then
    echo "MISS $base [$ext]: no golden (rerun with UPDATE=1)"
    miss=$((miss+1))
    rm -f "$tmp"
    return
  fi
  if diff -u "$exp" "$tmp" >/dev/null; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL $base [$ext]: output drift"
    diff -u "$exp" "$tmp" | sed 's/^/  /'
  fi
  rm -f "$tmp"
}

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  check_one "$m" "-emit-c"   "c.expected"
  check_one "$m" "-emit-cpp" "cpp.expected"
  # Flag-variant goldens: only checked when an opt-in golden file is
  # already present. Keeps the matrix manageable — flag behaviours are
  # locked in for the inputs that specifically exercise them.
  if [[ -e "${m%.m}.doxy.cpp.expected" ]]; then
    check_one "$m" "-emit-cpp -doxygen" "doxy.cpp.expected"
  fi
  if [[ -e "${m%.m}.auto.cpp.expected" ]]; then
    check_one "$m" "-emit-cpp -cpp-auto" "auto.cpp.expected"
  fi
done

echo "----"
echo "shape passed: $pass    failed: $fail    missing: $miss"
exit $(( fail > 0 || miss > 0 ? 1 : 0 ))
