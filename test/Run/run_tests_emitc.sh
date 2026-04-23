#!/usr/bin/env bash
# Build-and-run tests for the C/C++ emission path. For each .m in this
# directory, runs `matlabc -emit-c` (or -emit-cpp when MODE=cpp) and
# compiles the emitted source together with runtime/matlab_runtime.c using
# cc / c++. Compares stdout against the matching .stdout file. Failure if
# emission, compile, run, or diff fails.
#
# Usage: run_tests_emitc.sh <path-to-matlabc>
# Env:   MODE=c|cpp  (default: c)
#        CC=cc       C compiler (default: cc)
#        CXX=c++     C++ compiler (default: c++)
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

MODE="${MODE:-c}"
CC="${CC:-cc}"
CXX="${CXX:-c++}"

case "$MODE" in
  c)   FLAG="-emit-c";   COMPILE=("$CC" -w) ; EXT=c   ; LABEL="emit-c"   ;;
  cpp) FLAG="-emit-cpp"; COMPILE=("$CXX" -w) ; EXT=cpp ; LABEL="emit-cpp" ;;
  *)   echo "MODE must be c or cpp (got: $MODE)" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME="$ROOT/runtime/matlab_runtime.c"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"

pass=0; fail=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp="${m%.m}.stdout"
  [[ -e "$exp" ]] || { echo "SKIP $base (no .stdout)"; continue; }

  tmpsrc="$(mktemp -t mlc.XXXXXX).${EXT}"
  tmpbin="$(mktemp -t mlc.XXXXXX).out"

  if ! "$MATLABC" "$FLAG" "$m" > "$tmpsrc" 2>/dev/null; then
    echo "FAIL $base: matlabc $FLAG errored"
    fail=$((fail+1))
    rm -f "$tmpsrc" "$tmpbin"; continue
  fi

  # For C++ we need to compile the emitted file as C++ and the runtime as C
  # (the runtime is plain C). cc handles both in one invocation.
  if [[ "$MODE" == cpp ]]; then
    if ! "${COMPILE[@]}" -x c++ "$tmpsrc" -x c "$RUNTIME" -o "$tmpbin" \
           -lm -lpthread 2>/dev/null; then
      echo "FAIL $base: $LABEL compile failed"
      fail=$((fail+1))
      rm -f "$tmpsrc" "$tmpbin"; continue
    fi
  else
    if ! "${COMPILE[@]}" "$tmpsrc" "$RUNTIME" -o "$tmpbin" \
           -lm -lpthread 2>/dev/null; then
      echo "FAIL $base: $LABEL compile failed"
      fail=$((fail+1))
      rm -f "$tmpsrc" "$tmpbin"; continue
    fi
  fi

  got="$("$tmpbin")" || {
    echo "FAIL $base: non-zero exit"
    fail=$((fail+1))
    rm -f "$tmpsrc" "$tmpbin"; continue
  }

  if [[ -e "${m%.m}.sorted" ]]; then
    if diff -u <(sort "$exp") <(printf '%s\n' "$got" | sort) >/dev/null; then
      pass=$((pass+1))
    else
      fail=$((fail+1))
      echo "FAIL $base: stdout mismatch (sorted)"
      diff -u <(sort "$exp") <(printf '%s\n' "$got" | sort) | sed 's/^/  /'
    fi
  elif diff -u "$exp" <(printf '%s\n' "$got") >/dev/null; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL $base: stdout mismatch"
    diff -u "$exp" <(printf '%s\n' "$got") | sed 's/^/  /'
  fi
  rm -f "$tmpsrc" "$tmpbin"
done

echo "----"
echo "$LABEL passed: $pass    failed: $fail"
exit $(( fail > 0 ? 1 : 0 ))
