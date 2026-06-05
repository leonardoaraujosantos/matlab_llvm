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

# 5a. Cross-turn VideoWriter property-set (issue #236).  A `v = VideoWriter(...)`
#     in one turn followed by `v.FrameRate = ...` in a LATER turn used to route
#     the property store through the generic struct-field path (the new turn's
#     Lowerer had lost the VideoWriter classification), corrupting the opaque
#     handle and segfaulting in the encoder on the next writeVideo.  Each line
#     here is a separate REPL submission; the sentinel only prints if the split
#     setup left `v` usable.
run_case "xturn_videowriter_property_set" "$(cat <<'EOF'
v = VideoWriter('/tmp/repl_vw_236.mp4', 'MPEG-4');
v.FrameRate = 20;
open(v);
figure(1);
plot([0 1], [0 1]);
xlim([0 1]); ylim([0 1]);
writeVideo(v, getframe(gcf));
close(v);
disp('VW_XTURN_OK')
exit
EOF
)" "VW_XTURN_OK"
got_rc="$(printf '%s\n' "v = VideoWriter('/tmp/repl_vw_236.mp4', 'MPEG-4');" \
                       "v.FrameRate = 20;" "open(v);" "figure(1);" \
                       "plot([0 1],[0 1]); writeVideo(v, getframe(gcf));" \
                       "close(v);" "exit" | "$MATLABC" -repl >/dev/null 2>&1; \
          echo $?)"
if [[ "$got_rc" != "0" ]]; then
  fail=$((fail+1))
  fails+=("xturn_videowriter_property_set (exit rc=$got_rc, expected 0 — #236 handle corruption?)")
  echo "FAIL  xturn_videowriter_property_set (rc=$got_rc)"
fi

# 6. Cross-turn anonymous-handle persistence (issue #116).  An anonymous
#    function handle defined in one REPL turn must survive into the next:
#    before the fix, `f = @(x) ...` was kept on the local-slot lane and
#    emitted NO workspace store (capture-free anons were diverted past the
#    kind=13 persistence by #77), so a later turn read `f` back as an empty
#    matrix — yielding wrong values for a scalar call and a hard SIGBUS when
#    a solver invoked the empty pointer as a closure.  Named handles (`@sin`)
#    already persisted; these guard the anonymous case specifically.
run_case "xturn_anon_handle_scalar" "$(cat <<'EOF'
f = @(x) x + 1;
disp(f(5))
exit
EOF
)" "6"

# 6a. Zero-argument anon (`@() ...`) recovered + called across a turn — the
#     simplest capture-free handle, exercises the matlab_call_handle_s0 path.
run_case "xturn_anon_handle_zeroarg" "$(cat <<'EOF'
g = @() 42;
disp(g())
exit
EOF
)" "42"

# 6b. The headline crash: a capture-free anon defined one turn, passed to a
#     solver the next.  `fminunc` on the Rosenbrock banana converges to the
#     minimiser [1; 1]; the sentinel only prints if the cross-turn solver
#     call ran to completion (i.e. the handle round-tripped) — pre-fix this
#     SIGBUS'd and the substring was absent.
run_case "xturn_anon_handle_to_solver" "$(cat <<'EOF'
ros = @(x) 100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + (1 - x(1))*(1 - x(1));
r = fminunc(ros, [-1.2; 1]);
fprintf('ROS_OK %.0f %.0f\n', round(r(1)), round(r(2)));
exit
EOF
)" "ROS_OK 1 1"
# (Belt + braces: a SIGBUS would leak a shell signal and a non-zero rc even
#  though 2>&1 captured no sentinel.  Assert the process exits clean.)
got_rc="$(printf '%s\n' \
  'ros = @(x) 100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + (1 - x(1))*(1 - x(1));' \
  'r = fminunc(ros, [-1.2; 1]);' \
  'disp(r(1));' "exit" | "$MATLABC" -repl >/dev/null 2>&1; echo $?)"
if [[ "$got_rc" != "0" ]]; then
  fail=$((fail+1))
  fails+=("xturn_anon_handle_to_solver (exit rc=$got_rc, expected 0 — SIGBUS regression?)")
  echo "FAIL  xturn_anon_handle_to_solver (rc=$got_rc)"
fi

# 7. Cross-turn handle called DIRECTLY with a MATRIX argument (issue #119).
#    Before the fix the recovered kind=13 pointer was subscripted as a
#    matrix — SIGSEGV (and, when the garbage dimension was huge, a multi-GB
#    runaway allocation).  Scalar-return objective shape:
run_case "xturn_anon_handle_matrix_arg_scalar" "$(cat <<'EOF'
f = @(x) x(1) + x(2);
v = [3; 4];
disp(f(v))
exit
EOF
)" "7"

# 7b. Matrix-RETURN handle called cross-turn with a matrix arg (the
#     vector->vector residual shape — dispatches to matlab_call_handle_mm*).
run_case "xturn_anon_handle_matrix_arg_matrix" "$(cat <<'EOF'
r = @(x) [x(1); x(2) * 2];
v = [3; 4];
s = r(v);
fprintf('MM %.0f %.0f\n', s(1), s(2));
exit
EOF
)" "MM 3 8"

# 7c. The headline #119 flow + no-runaway guard: define an objective, run a
#     solver on it (cross-turn), THEN call the handle directly with the
#     solver's matrix result.  Asserts a clean exit — a non-zero rc here
#     means either the SIGSEGV came back or the bogus-dimension allocation
#     blew up (the multi-GB hang the fix removes).
got_rc="$(printf '%s\n' \
  'rastrigin = @(x) 20 + (x(1)*x(1) - 10*cos(2*pi*x(1))) + (x(2)*x(2) - 10*cos(2*pi*x(2)));' \
  'xs = fminunc(rastrigin, [3.1; 2.9]);' \
  'fprintf("f=%.4f\n", rastrigin(xs));' "exit" | "$MATLABC" -repl >/dev/null 2>&1; echo $?)"
if [[ "$got_rc" != "0" ]]; then
  fail=$((fail+1))
  fails+=("xturn_anon_handle_matrix_arg_solver (exit rc=$got_rc, expected 0 — SIGSEGV / runaway regression?)")
  echo "FAIL  xturn_anon_handle_matrix_arg_solver (rc=$got_rc)"
fi

# 8. Cross-turn table persistence (issue #116). A table defined in one REPL
#    turn must round-trip with its type intact: before the fix the kind=6
#    workspace value read back untyped (the Resolver never stamped the
#    binding as a table), so a later `height(T)` / `width(T)` hit
#    "unsupported call shape" and `disp(T)` / `T.col` crashed. The Resolver
#    now stamps Binding::IsTable on the kind=6 lookup and the lowering
#    re-populates TableBindings.
run_case "xturn_table_height_width" "$(cat <<'EOF'
T = table([10; 20; 30], [1; 2; 3]);
fprintf('TBL %.0f %.0f\n', height(T), width(T));
exit
EOF
)" "TBL 3 2"
# (Belt + braces: a use-site crash would leak a signal / non-zero rc.)
got_rc="$(printf '%s\n' \
  'T = table([10; 20; 30], [1; 2; 3]);' \
  'disp(height(T)); disp(T.Var1);' "exit" | "$MATLABC" -repl >/dev/null 2>&1; echo $?)"
if [[ "$got_rc" != "0" ]]; then
  fail=$((fail+1))
  fails+=("xturn_table_height_width (exit rc=$got_rc, expected 0 — table round-trip crash?)")
  echo "FAIL  xturn_table_height_width (rc=$got_rc)"
fi
# 9. Cross-turn struct field-assignment persistence (issue #131). A struct
#    created/mutated by a field store (`s.x = v`) must round-trip the
#    workspace: before the fix the field write went only to s's local slot
#    and emitted no matlab_ws_set_struct, so `s` never entered the workspace
#    and a later-turn `s.x` read an empty struct -> 0.
run_case "xturn_struct_field" "$(cat <<'EOF'
s.x = 42;
disp(s.x)
exit
EOF
)" "42"

# 9b. Nested field chain (`s.a.b = v`) — the root struct must persist too.
run_case "xturn_struct_field_nested" "$(cat <<'EOF'
s.a.b = 7;
disp(s.a.b)
exit
EOF
)" "7"
# 8. Cross-turn struct-array persistence (issue #133). A struct array built by
#    indexed field-assignment (`a(i).x = v`) must round-trip the workspace:
#    before the fix the array lived only in a local slot (no
#    matlab_ws_set_struct_arr), so `a` never entered the workspace and a
#    later-turn `a(i).x` / `length(a)` read an empty array. Distinct from the
#    plain-struct case (#131): struct arrays use matlab_struct_arr* (kind=14).
run_case "xturn_struct_array_field" "$(cat <<'EOF'
a(1).x = 5;
a(2).x = 9;
disp(a(2).x)
exit
EOF
)" "9"

# 8b. length() of a cross-turn struct array.
run_case "xturn_struct_array_length" "$(cat <<'EOF'
a(1).x = 5;
a(2).x = 9;
disp(length(a))
exit
EOF
)" "2"

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  echo "failures:"
  for f in "${fails[@]}"; do echo "  - $f"; done
fi
exit $(( fail > 0 ? 1 : 0 ))
