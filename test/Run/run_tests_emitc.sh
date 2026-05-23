#!/usr/bin/env bash
# Build-and-run tests for the C/C++ emission path. For each .m in this
# directory, runs `matlabc -emit-c` (or -emit-cpp when MODE=cpp) and
# compiles the emitted source together with runtime/matlab_runtime.c using
# cc / c++. Compares stdout against the matching .stdout file. Failure if
# emission, compile, run, or diff fails.
#
# Usage: run_tests_emitc.sh <path-to-matlabc>
# Env:   MODE=c|cpp  (default: c)
#        STRICT=1    compile with -Wall -Wextra -Werror instead of -w
#        CC=cc       C compiler (default: cc)
#        CXX=c++     C++ compiler (default: c++)
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

MODE="${MODE:-c}"
STRICT="${STRICT:-0}"
CC="${CC:-cc}"
CXX="${CXX:-c++}"
# The runtime is a C++20 project (CMAKE_CXX_STANDARD 20); compile its TUs with
# that standard so libstdc++ pulls in <string>/<cstdio> (the compiler default
# gnu++17 leaves them incomplete on Linux).  The emitted source side keeps the
# default standard (it only includes the public headers).
CXXSTD="${CXXSTD:--std=c++20}"

# In strict mode: treat warnings as errors, but exempt categories that are
# inherent to the emitter's output shape (one C local per SSA value, so
# lots of writes-without-use in dead branches) or to the runtime's macros
# (unused induction vars inside COLWISE_REDUCE). Everything else — type
# confusion, implicit decls, sign mismatches, missing returns — must pass.
if [[ "$STRICT" == "1" ]]; then
  WFLAGS=(-Wall -Wextra -Werror
          -Wno-unused-variable -Wno-unused-but-set-variable
          -Wno-unused-parameter -Wno-unused-function
          -Wno-parentheses-equality
          # Designated-initializer structs (e.g. matlab_dbg_state) omit
          # zero-defaulted trailing fields on purpose; clang's -Wextra flags
          # this under libstdc++ but it is not a real defect.
          -Wno-missing-field-initializers)
  LABEL_SUFFIX=" strict"
else
  WFLAGS=(-w)
  LABEL_SUFFIX=""
fi

case "$MODE" in
  c)   FLAG="-emit-c";   COMPILE=("$CC"  "${WFLAGS[@]}"); EXT=c   ; LABEL="emit-c${LABEL_SUFFIX}"   ;;
  cpp) FLAG="-emit-cpp"; COMPILE=("$CXX" "${WFLAGS[@]}"); EXT=cpp ; LABEL="emit-cpp${LABEL_SUFFIX}" ;;
  *)   echo "MODE must be c or cpp (got: $MODE)" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Runtime is C++ since Phase 3 of docs/port_runtime_2_cpp.md. The C
# emit-c path still emits valid C but links against the C++ runtime;
# we force the runtime side through the C++ compiler with -x c++.
# Phase-2 + 2.5 split: three .cpp files share private layouts.
RUNTIME_MAIN="$ROOT/runtime/matlab_runtime.cpp"
RUNTIME_DEBUG="$ROOT/runtime/runtime_debug.cpp"
RUNTIME_COMPLEX="$ROOT/runtime/runtime_complex.cpp"
RUNTIME_COMM="$ROOT/runtime/toolbox/comm/runtime_comm.cpp"
RUNTIME_PROP="$ROOT/runtime/toolbox/prop/runtime_prop.cpp"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"

# Precompile the runtime TUs once (they used to be recompiled per fixture,
# which dominated this lane's wall time). The emitted source still compiles
# per fixture — only the fixed runtime objects are cached and linked in. The
# strict WFLAGS are applied here too, so a runtime warning still fails the
# strict lanes.
OBJDIR="$(mktemp -d -t mlc-emitc.XXXXXX)"
trap 'rm -rf "$OBJDIR"' EXIT
RUNTIME_OBJS=()
for src in "$RUNTIME_MAIN" "$RUNTIME_DEBUG" "$RUNTIME_COMPLEX" "$RUNTIME_COMM" "$RUNTIME_PROP"; do
  obj="$OBJDIR/$(basename "${src%.cpp}").o"
  if ! "$CXX" $CXXSTD "${WFLAGS[@]}" "-I$ROOT/runtime" -x c++ -c "$src" -o "$obj" 2>"$OBJDIR/cc.err"; then
    echo "FATAL: failed to compile runtime TU $src" >&2
    cat "$OBJDIR/cc.err" >&2
    exit 2
  fi
  RUNTIME_OBJS+=( "$obj" )
done

pass=0; fail=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp="${m%.m}.stdout"
  [[ -e "$exp" ]] || { echo "SKIP $base (no .stdout)"; continue; }
  # Per-mode skip files. Programs that compile / run cleanly under
  # LLVM and one of the C/C++ lanes but trip a known bug in the other
  # ship a `.skip-emit-c` or `.skip-emit-cpp` next to the .m to mark
  # the lane carve-out (matches the existing .skip-emit-python /
  # .skip-emit-typescript convention).
  if [[ "$MODE" == cpp && -e "${m%.m}.skip-emit-cpp" ]]; then
    echo "SKIP $base (marked .skip-emit-cpp)"; continue
  fi
  if [[ "$MODE" == c && -e "${m%.m}.skip-emit-c" ]]; then
    echo "SKIP $base (marked .skip-emit-c)"; continue
  fi

  tmpsrc="$(mktemp -t mlc.XXXXXX).${EXT}"
  tmpbin="$(mktemp -t mlc.XXXXXX).out"

  if ! "$MATLABC" "$FLAG" "$m" > "$tmpsrc" 2>/dev/null; then
    echo "FAIL $base: matlabc $FLAG errored"
    fail=$((fail+1))
    rm -f "$tmpsrc" "$tmpbin"; continue
  fi

  # The runtime is C++ (Phase 3 of the runtime port). For MODE=c the
  # emitted file is C and the runtime side needs the C++ compiler — drive
  # the link line with $CXX in both modes, forcing the input language
  # explicitly with -x.
  cc_err="$(mktemp -t mlc.XXXXXX).err"
  # The emitted file is C (MODE=c) or C++ (MODE=cpp); the runtime objects are
  # already compiled. Compile the emitted source and link the cached objects.
  xlang=c++; [[ "$MODE" == c ]] && xlang=c
  # `-x none` after the emitted source resets language detection so the
  # prebuilt .o objects are treated as objects (not as `-x $xlang` source).
  if ! "$CXX" "${WFLAGS[@]}" "-I$ROOT/runtime" -x "$xlang" "$tmpsrc" \
         -x none "${RUNTIME_OBJS[@]}" \
         -o "$tmpbin" -lm -lpthread 2>"$cc_err"; then
    echo "FAIL $base: $LABEL compile failed"
    [[ "$STRICT" == "1" ]] && sed 's/^/  /' "$cc_err" | head -5
    fail=$((fail+1))
    rm -f "$tmpsrc" "$tmpbin" "$cc_err"; continue
  fi
  rm -f "$cc_err"

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
