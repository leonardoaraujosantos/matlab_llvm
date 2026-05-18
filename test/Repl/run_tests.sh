#!/usr/bin/env bash
# matlabc REPL smoke tests.
#
# Each case feeds a scripted REPL conversation over stdin and matches
# the stdout against an expected substring set. The headline coverage
# is cross-turn user-function persistence: a `function ... end` block
# submitted in REPL turn N is stashed in g_ReplUserFunctions and
# pulled back via buildReplPrelude on turn N+1, so the function can
# be called interactively.
#
# Usage: run_tests.sh <path-to-matlabc>
set -u

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

pass=0; fail=0
fails=()

run_case() {
  local name="$1" input="$2" want="$3"
  local got
  got="$(printf '%s\n' "$input" | "$MATLABC" -repl 2>&1)"
  if grep -qF "$want" <<<"$got"; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    fails+=("$name (expected substring '$want' missing from output)")
    echo "FAIL  $name"
    echo "----- got -----"
    echo "$got"
    echo "----- want substring -----"
    echo "$want"
    echo "---------------"
  fi
}

# 1. Cross-turn user function: define in one block, call in another.
run_case "cross_turn_user_fn" "$(cat <<'EOF'
function y = double_it(x)
y = 2 * x;
end
disp(double_it(5))
exit
EOF
)" "10"

# 2. Transitive cross-turn: f calls g, both defined in prior blocks.
run_case "transitive_user_fns" "$(cat <<'EOF'
function y = add(a, b)
y = a + b;
end
function z = quad(x)
z = add(x*x, 2*x) + 1;
end
disp(quad(3))
exit
EOF
)" "16"

# 3. Redefinition: a name re-defined in a turn replaces the stashed
#    copy without colliding with the prelude.
run_case "redef_user_fn" "$(cat <<'EOF'
function y = f(x)
y = x + 1;
end
disp(f(10))
function y = f(x)
y = x * 2;
end
disp(f(10))
exit
EOF
)" "20"

# 4. Function defined but never called: no error message printed.
run_case "uncalled_user_fn" "$(cat <<'EOF'
function y = unused(x)
y = x + 1;
end
exit
EOF
)" "matlabc REPL"
# (Negative check: should NOT contain the func.func translation error.)
got="$(printf "function y = unused(x)\ny = x + 1;\nend\nexit\n" | \
       "$MATLABC" -repl 2>&1)"
if grep -q "missing.*LLVMTranslationDialectInterface" <<<"$got"; then
  fail=$((fail+1))
  fails+=("uncalled_user_fn (unexpected JIT translation error)")
  echo "FAIL  uncalled_user_fn"
  echo "$got"
fi

# 5. Plot-handle assignment: `hSurf = surfc(X, Y, Z)` previously
#    crashed the REPL JIT with status 11 (SIGSEGV) because the
#    void-rewritten call_builtin left a dangling
#    `matlab_ws_set_obj` user that tripped the MLIR verifier when it
#    formatted the dangling op. The disp sentinel only prints if the
#    statement after the surf-with-assignment compiled and ran
#    successfully.
run_case "plot_handle_assign_no_crash" "$(cat <<'EOF'
[X, Y, Z] = peaks(8);
hSurf = surfc(X, Y, Z);
disp('SURFC_ASSIGN_OK')
hMesh = mesh(X, Y, Z);
disp('MESH_ASSIGN_OK')
hSurf2 = surfl(X, Y, Z);
disp('SURFL_ASSIGN_OK')
exit
EOF
)" "SURFL_ASSIGN_OK"
# (Cross-check: the harness pipes 2>&1, so a runtime crash would
# leak shell "Segmentation fault" output into $got and the substring
# check above would miss it.  Belt + braces: rerun + capture exit
# rc, fail if non-zero.)
got_rc="$(printf '%s\n' '[X, Y, Z] = peaks(8);' 'hSurf = surfc(X, Y, Z);' \
                       "exit" | "$MATLABC" -repl >/dev/null 2>&1; \
          echo $?)"
if [[ "$got_rc" != "0" ]]; then
  fail=$((fail+1))
  fails+=("plot_handle_assign_no_crash (exit rc=$got_rc, expected 0)")
  echo "FAIL  plot_handle_assign_no_crash (rc=$got_rc)"
fi

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  echo "failures:"
  for f in "${fails[@]}"; do echo "  - $f"; done
fi
exit $(( fail > 0 ? 1 : 0 ))
