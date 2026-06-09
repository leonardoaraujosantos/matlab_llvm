#!/usr/bin/env bash
# Known interactive-REPL (-repl) cross-turn defects — xfail tracker.
#
# Each case is a MINIMAL reproduction of an open bug under #116 (the REPL
# workspace round-trip is not fully kind/shape/class-preserving).  This is NOT
# part of the main gate: it ASSERTS THE BUG IS STILL PRESENT (an "expected
# failure" marker substring appears in the -repl output).  When a fix lands and
# a repro stops emitting its marker, this script reports the case as
# FIXED and exits non-zero — a loud reminder to (1) close the linked issue,
# and (2) promote the repro into a real passing regression in run_tests.sh.
#
# Usage: known_issues.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

xfail=0      # bug reproduced (still broken) — expected
fixed=0      # bug no longer reproduces — needs attention
fixed_names=()

# xfail_case NAME ISSUE INPUT MARKER
#   Runs INPUT through `matlabc -repl`; the bug is "still present" iff MARKER
#   (a fixed substring) appears in the combined output.
xfail_case() {
  local name="$1" issue="$2" input="$3" marker="$4"
  local got
  got="$(printf '%s\n' "$input" | "$MATLABC" -repl 2>&1)"
  if grep -qF "$marker" <<<"$got"; then
    xfail=$((xfail+1))
    echo "xfail  $name (#$issue) — still reproduces"
  else
    fixed=$((fixed+1))
    fixed_names+=("$name (#$issue)")
    echo "FIXED? $name (#$issue) — marker '$marker' gone; promote to a passing regression and close the issue"
  fi
}

# --- #258: struct/struct-array field read loses its matrix type cross-turn ---
xfail_case "xturn_struct_field_matrix_type" 258 "$(cat <<'EOF'
s.Payload = [1 1 0 0];
a = [1 0 1 0];
disp(biterr(a, s.Payload))
exit
EOF
)" "unsupported call shape for built-in function 'biterr'"

# (#259 timetable summary cross-turn — FIXED, promoted to run_tests.sh::xturn_timetable_summary)

# --- #260: REPL does not accumulate a multi-line classdef block ---
xfail_case "repl_classdef_block_accumulation" 260 "$(cat <<'EOF'
classdef Pt
 properties
  x
 end
end
exit
EOF
)" "'end' is only valid inside indexing"

# --- #261: dlnet dlarray pin lost on in-loop reassignment + reshape operand ---
xfail_case "xturn_dlarray_reassign_reshape_matmul" 261 "$(cat <<'EOF'
W2_dl = dlarray(ones(3,16));
for k = 1:2
    logits = W2_dl * reshape(dlarray(ones(2,2,4,4)), 16, 4);
    yhat   = softmax(logits);
    W2_dl  = dlarray(ones(3,16) - 0.2);
end
disp(99)
exit
EOF
)" "unsupported call shape for built-in function 'softmax'"

echo "----"
echo "known-issue xfails: $xfail still-broken, $fixed now-fixed"
if (( fixed > 0 )); then
  printf '  needs promotion: %s\n' "${fixed_names[@]}"
  exit 1
fi
exit 0
