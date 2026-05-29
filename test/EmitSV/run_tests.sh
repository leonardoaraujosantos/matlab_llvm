#!/usr/bin/env bash
# Golden tests for the -emit-systemverilog backend (ASIC, Phase 1).
#
# For every .m in this directory, runs `matlabc -emit-systemverilog`
# and diffs stdout against the matching .sv.expected golden. When
# Verilator is on PATH (or VERILATOR is set), each emitted .sv is
# additionally lint-checked with `verilator --lint-only -Wall` —
# catches inferred-latch hazards and missing-width literals that the
# golden-diff lane alone cannot see.
#
# Usage: run_tests.sh <path-to-matlabc>
# Env:
#   UPDATE=1     regenerate every .sv.expected
#   VERILATOR=…  override or skip the lint pass (set empty to skip)
#                — defaults to the `verilator` on PATH if any
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

UPDATE="${UPDATE:-0}"
# Bless-size guard. When UPDATE=1 rewrites the goldens, accumulate
# the before/after total line counts; if the delta exceeds 5% of
# the original total, print a warning so a careless `UPDATE=1` run
# can't silently drift the corpus. Set BLESS_LIMIT=N to override
# the threshold (percent), or BLESS_LIMIT=off to disable the
# guard.
BLESS_LIMIT="${BLESS_LIMIT:-5}"
bless_old_lines=0
bless_new_lines=0
VERILATOR="${VERILATOR-$(command -v verilator || true)}"
# Yosys is an optional second-opinion synth lint that complements
# Verilator's `-Wall` style checks: it parses each emitted module
# through a real synthesis frontend and runs the `synth` flow,
# which catches inferable-latch hazards, multi-driver nets, and
# elaboration-time constructs Verilator is lenient about. Yosys
# 0.x's SV frontend has limited LRM coverage (e.g. `'{default: '0}`
# array literals trip it), so the lint is informational by
# default — failures are tallied separately and don't gate the
# suite. Set YOSYS_STRICT=1 to make Yosys-detected failures count
# as suite failures (useful in CI once the corpus is curated).
YOSYS="${YOSYS-$(command -v yosys || true)}"
YOSYS_STRICT="${YOSYS_STRICT:-0}"
TESTDIR="$(cd "$(dirname "$0")" && pwd)"
pass=0; fail=0; miss=0; lint_skip=0
yosys_pass=0; yosys_skip=0; yosys_fail=0

for m in "$TESTDIR"/*.m; do
  [[ -e "$m" ]] || continue
  base="$(basename "${m%.m}")"
  exp="${m%.m}.sv.expected"
  # Verilator's DECLFILENAME warning fires whenever the on-disk
  # filename doesn't match the top module name; using `mktemp` would
  # produce `mlc.XXXXXX.sv` and trip every lint. Drop into a per-test
  # tempdir whose file is `<base>.sv`.
  tmpdir="$(mktemp -d -t emitsv.XXXXXX)"
  tmp="$tmpdir/$base.sv"

  if ! "$MATLABC" -emit-systemverilog "$m" > "$tmp" 2>/dev/null; then
    echo "FAIL $base: matlabc -emit-systemverilog exited non-zero"
    fail=$((fail+1))
    rm -rf "$tmpdir"
    continue
  fi

  if [[ "$UPDATE" == "1" ]]; then
    if [[ -e "$exp" ]]; then
      bless_old_lines=$(( bless_old_lines + $(wc -l < "$exp") ))
    fi
    bless_new_lines=$(( bless_new_lines + $(wc -l < "$tmp") ))
    mv "$tmp" "$exp"
    echo "UPDATE $base"
    pass=$((pass+1))
    continue
  fi

  if [[ ! -e "$exp" ]]; then
    echo "MISS $base: no golden (rerun with UPDATE=1)"
    miss=$((miss+1))
    rm -rf "$tmpdir"
    continue
  fi

  if ! diff -u "$exp" "$tmp" >/dev/null; then
    echo "FAIL $base: golden drift"
    diff -u "$exp" "$tmp" | sed 's/^/  /'
    fail=$((fail+1))
    rm -rf "$tmpdir"
    continue
  fi

  # Verilator lint smoke. Top module name = file basename so the
  # DECLFILENAME warning is silenced when they match. -Wno-DECLFILENAME
  # additionally suppresses the warning for hierarchical fixtures
  # where multiple modules share the file (the convention of
  # one-module-per-file is a verilator preference, not an SV
  # requirement; HDL workflow tools split modules at synth time).
  if [[ -n "$VERILATOR" && -x "$VERILATOR" ]]; then
    # Per-fixture `<name>.lint-extra` adds extra verilator flags (e.g.
    # `-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC`) for fixtures that use the
    # `% hdl: precise_fi` opt-in -- their FL-grown intermediates +
    # narrow output port legitimately trip the width-narrowing
    # warnings, even though the math is bit-accurate vs. the Python
    # reference (issue #75).
    extra_lint=""
    if [[ -f "${m%.m}.lint-extra" ]]; then
      extra_lint="$(cat "${m%.m}.lint-extra")"
    fi
    if ! "$VERILATOR" --lint-only -Wall -Wno-DECLFILENAME $extra_lint --top-module "$base" "$tmp" >/dev/null 2>&1; then
      echo "FAIL $base: verilator lint"
      "$VERILATOR" --lint-only -Wall -Wno-DECLFILENAME $extra_lint --top-module "$base" "$tmp" 2>&1 | sed 's/^/  /'
      fail=$((fail+1))
      rm -rf "$tmpdir"
      continue
    fi
  else
    lint_skip=$((lint_skip+1))
  fi

  # Yosys synth-flow lint (informational). Yosys's SV frontend can
  # reject constructs Verilator accepts (most commonly the
  # `'{default: '0}` array literal — Yosys 0.x doesn't parse it).
  # Capture the result without gating: track pass / skip / fail so
  # the summary line surfaces coverage without breaking the suite
  # when a fixture trips a known frontend gap. YOSYS_STRICT=1
  # promotes failures into suite failures.
  if [[ -n "$YOSYS" && -x "$YOSYS" ]]; then
    if "$YOSYS" -p "read_verilog -sv $tmp; synth -top $base" \
         > "$tmpdir/yosys.log" 2>&1; then
      yosys_pass=$((yosys_pass+1))
    else
      if grep -qE "syntax error|Unknown.*type|Failed to" "$tmpdir/yosys.log"; then
        yosys_skip=$((yosys_skip+1))
      else
        yosys_fail=$((yosys_fail+1))
        if [[ "$YOSYS_STRICT" == "1" ]]; then
          echo "FAIL $base: yosys synth"
          tail -10 "$tmpdir/yosys.log" | sed 's/^/  /'
          fail=$((fail+1))
          rm -rf "$tmpdir"
          continue
        fi
      fi
    fi
  fi

  pass=$((pass+1))
  rm -f "$tmp"
done

# Phase 4 v2.5 — quick smoke sweep of the FSM-encoding flag on
# every FSM fixture (anything whose filename starts with fsm_ /
# mealy_ / moore_). For each, emit with -sv-fsm-encoding=one-hot
# and -sv-fsm-encoding=gray and re-lint with Verilator. We don't
# diff goldens for these — only the typedef enum line changes
# and that's already covered by the binary golden — but we do
# verify the alternate encodings still produce lint-clean SV.
enc_pass=0; enc_fail=0
if [[ -n "$VERILATOR" && -x "$VERILATOR" ]]; then
  for m in "$TESTDIR"/*.m; do
    base="$(basename "${m%.m}")"
    case "$base" in
      fsm_*|mealy_*|moore_*) ;;
      *) continue ;;
    esac
    for enc in one-hot gray; do
      tmpdir="$(mktemp -d -t emitsvenc.XXXXXX)"
      tmp="$tmpdir/$base.sv"
      if ! "$MATLABC" -emit-systemverilog -sv-fsm-encoding="$enc" \
           "$m" > "$tmp" 2>/dev/null; then
        echo "FAIL $base [enc=$enc]: matlabc exited non-zero"
        enc_fail=$((enc_fail+1))
        rm -rf "$tmpdir"; continue
      fi
      if ! "$VERILATOR" --lint-only -Wall --top-module "$base" \
           "$tmp" >/dev/null 2>&1; then
        echo "FAIL $base [enc=$enc]: verilator lint"
        "$VERILATOR" --lint-only -Wall --top-module "$base" "$tmp" 2>&1 \
          | sed 's/^/  /'
        enc_fail=$((enc_fail+1))
      else
        enc_pass=$((enc_pass+1))
      fi
      rm -rf "$tmpdir"
    done
  done
fi

echo "----"
# Bless-size guard report (UPDATE=1 only).
if [[ "$UPDATE" == "1" && "$BLESS_LIMIT" != "off" && $bless_old_lines -gt 0 ]]; then
  delta=$(( bless_new_lines - bless_old_lines ))
  abs_delta=${delta#-}
  pct=$(( abs_delta * 100 / bless_old_lines ))
  if [[ $pct -gt $BLESS_LIMIT ]]; then
    echo "WARNING: golden corpus changed by ${pct}% (was $bless_old_lines lines, now $bless_new_lines, delta $delta) — review the diffs manually before committing. Set BLESS_LIMIT=$pct to silence." >&2
  fi
fi
if [[ $lint_skip -gt 0 ]]; then
  echo "emit-sv passed: $pass    failed: $fail    missing: $miss    (verilator skipped on $lint_skip)"
else
  echo "emit-sv passed: $pass    failed: $fail    missing: $miss    (verilator lint included)"
fi
if [[ $enc_pass -gt 0 || $enc_fail -gt 0 ]]; then
  echo "fsm-encoding sweep: passed: $enc_pass    failed: $enc_fail"
fi
if [[ -n "$YOSYS" ]] && [[ $yosys_pass -gt 0 || $yosys_skip -gt 0 || $yosys_fail -gt 0 ]]; then
  if [[ "$YOSYS_STRICT" == "1" ]]; then
    echo "yosys synth: passed: $yosys_pass    skipped: $yosys_skip    failed: $yosys_fail (strict)"
  else
    echo "yosys synth: passed: $yosys_pass    skipped: $yosys_skip    failed: $yosys_fail (informational)"
  fi
fi
exit $(( fail > 0 || miss > 0 || enc_fail > 0 ? 1 : 0 ))
