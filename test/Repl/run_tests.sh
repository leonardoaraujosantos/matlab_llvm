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

# Assert a substring is ABSENT from the -repl output (e.g. a parse error that a
# fix must no longer emit).  The REPL is lenient — it recovers and keeps going
# after an error — so some fixes (e.g. classdef block accumulation) have no
# positive output to match; their only observable is the absence of the error.
run_case_absent() {
  local name="$1" input="$2" forbidden="$3"
  local got
  got="$(printf '%s\n' "$input" | "$MATLABC" -repl 2>&1)"
  if grep -qF "$forbidden" <<<"$got"; then
    fail=$((fail+1))
    fails+=("$name (forbidden substring '$forbidden' present in output)")
    echo "FAIL  $name"
    echo "----- got -----"
    echo "$got"
    echo "----- forbidden substring -----"
    echo "$forbidden"
    echo "---------------"
  else
    pass=$((pass+1))
  fi
}

# Like run_case, but launches the REPL from <dir> so current-folder `.m`
# function files are in scope (issue #284). MODE "want"/"absent" selects
# whether the substring must be present or must NOT appear.
run_case_dir() {
  local dir="$1" name="$2" input="$3" mode="$4" sub="$5"
  local got
  got="$(cd "$dir" && printf '%s\n' "$input" | "$MATLABC" -repl 2>&1)"
  local ok=1
  if [[ "$mode" == "absent" ]]; then
    grep -qF "$sub" <<<"$got" && ok=0
  else
    grep -qF "$sub" <<<"$got" || ok=0
  fi
  if (( ok )); then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    fails+=("$name (substring '$sub' ${mode}-check failed)")
    echo "FAIL  $name"
    echo "----- got -----"; echo "$got"
    echo "----- ${mode} substring -----"; echo "$sub"
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

# 3a. #290 — multi-line control blocks entered across several input lines.
#     The REPL accumulates lines until block/bracket depth returns to 0 (and
#     past a trailing `...` continuation), then compiles+runs the whole block
#     as one unit. These lock that behavior for for/if/while/switch/try, a
#     nested block, a multi-line function def + call, and a `...` continuation.
run_case "multiline_for_block" "$(cat <<'EOF'
for i = 1:3
  disp(i)
end
exit
EOF
)" "3"
run_case "multiline_if_else_block" "$(cat <<'EOF'
x = 5;
if x > 3
  disp('big')
else
  disp('small')
end
exit
EOF
)" "big"
run_case "multiline_while_block" "$(cat <<'EOF'
k = 0;
while k < 3
  k = k + 1;
end
disp(k)
exit
EOF
)" "3"
run_case "multiline_switch_block" "$(cat <<'EOF'
x = 2;
switch x
  case 1
    disp('one')
  case 2
    disp('two')
end
exit
EOF
)" "two"
run_case "multiline_try_catch_block" "$(cat <<'EOF'
try
  error('boom')
catch e
  disp('caught')
end
exit
EOF
)" "caught"
run_case "multiline_nested_blocks" "$(cat <<'EOF'
for i = 1:2
  for j = 1:2
    if i == j
      disp(i*10 + j)
    end
  end
end
exit
EOF
)" "22"
run_case "multiline_line_continuation" "$(cat <<'EOF'
v = [1 2 ...
3 4];
disp(sum(v))
exit
EOF
)" "10"

# 3b. #286 — a bare, non-suppressed user-function call auto-displays its
#     result as `ans = <value>` (it previously ran but printed nothing, while
#     builtins displayed). Covers cross-turn scalar, the matrix-returning
#     case, and that `ans` is reusable in a later turn.
run_case "bare_user_fn_autodisplay_ans" "$(cat <<'EOF'
function y = f(x)
y = x + 2;
end
f(40)
exit
EOF
)" "ans ="
run_case "bare_user_fn_autodisplay_value" "$(cat <<'EOF'
function y = f(x)
y = x + 2;
end
f(40)
exit
EOF
)" "42"
# Result binds to `ans` and is reusable on a later turn.
run_case "bare_user_fn_ans_reuse" "$(cat <<'EOF'
function y = f(x)
y = x + 2;
end
f(40)
ans + 1
exit
EOF
)" "43"
# Matrix-returning bare call displays the matrix (not a scalar mis-disp).
run_case "bare_user_fn_matrix_autodisplay" "$(cat <<'EOF'
function M = mk()
M = [1 2; 3 4];
end
mk()
exit
EOF
)" "ans ="
# Suppressed bare call must NOT auto-display.
run_case_absent "bare_user_fn_suppressed_no_display" "$(cat <<'EOF'
function y = f(x)
y = x + 2;
end
f(40);
exit
EOF
)" "ans ="

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

# 8c. #116 — a 3-D array (matlab_mat3) round-trips through the workspace under
# the generic mat kind, so a cross-turn N-D subscript store/read used to back
# off to "unsupported call shape for __subscript_store".  The Resolver now
# stamps Binding::IsThreeD from the kind=16 workspace lookup and the lowering
# re-seeds the 3-D dispatch (isThreeDBinding), so the plane store + element
# read work across turns.  Without the fix turn 2 fails to compile, `img`
# stays all-zero, and the read prints 0 instead of 7.
run_case "xturn_threed_plane_store" "$(cat <<'EOF'
img = zeros(2, 2, 3);
img(:, :, 2) = 7;
disp(img(1, 1, 2))
exit
EOF
)" "7"

# 8d. #116 — a classdef instance (here occupancyMap) used across >=3 turns.
# ClassId is a per-TU positional counter, so the obj's stored class_id maps to
# a DIFFERENT class on a later turn (the registry slot is overwritten), and the
# Resolver fails to re-pin the binding -> a 3rd-turn `getOccupancy`/`setOccupancy`
# backs off to "unsupported call shape".  The fix captures the class NAME at
# obj-store time (matlab_ws_set_obj) and resolves cross-turn by name, immune to
# id reassignment.  Without it, turn 3 errors and prints nothing instead of 1.
run_case "xturn_obj_class_repin_3turn" "$(cat <<'EOF'
m = occupancyMap(10, 10);
setOccupancy(m, [1 1], 1);
disp(getOccupancy(m, [1 1]))
exit
EOF
)" "1"

# 8e. #259 — a timetable round-trips generically, so a cross-turn `summary(TT)`
# used to be mis-typed as a plain matrix and back off ("unsupported call shape").
# The fix marks the var as a timetable at store time (matlab_ws_mark_timetable),
# the kind hook reports a distinct kind, and the Resolver re-stamps
# Binding::IsTimetable so summary/head/TT.col route to the timetable path.
run_case "xturn_timetable_summary" "$(cat <<'EOF'
TT = timetable([1;2;3], [10;20;30]);
summary(TT);
exit
EOF
)" "Var2: NumMissing=0  Min=10  Max=30  Mean=20"

# 8f. #260 (Symptom A) — a multi-line classdef block fed to -repl must be
# accumulated as ONE submission.  blockDepth() now counts properties/methods/
# events/enumeration as block-openers inside a classdef; without that, the
# accumulator submitted a partial `classdef ... properties ... end` after the
# *properties* `end` and the leftover classdef `end` parsed standalone as
# `'end' is only valid inside indexing`.  Absence-assertion: that error must
# no longer appear.
run_case_absent "repl_classdef_block_accumulation" "$(cat <<'EOF'
classdef Pt
 properties
  x
 end
end
disp(7)
exit
EOF
)" "'end' is only valid inside indexing"

# 8g. #258 — a struct round-trips into a later REPL turn without its per-field
# element-kind, so a matrix field (`s.Payload`) defaulted to a scalar get_f64
# and a builtin consuming it (`biterr(a, s.Payload)`) backed off to
# "unsupported call shape".  A cross-turn struct-field read now fetches via
# get_mat (kind-aware; boxes a true scalar to 1x1).
run_case "xturn_struct_field_matrix" "$(cat <<'EOF'
s.Payload = [1 1 0 0];
a = [1 0 1 0];
disp(biterr(a, s.Payload))
exit
EOF
)" "0.5"

# 8h. #258 (struct-array variant) — a struct ARRAY field `s(i).Field` loses its
# element-kind cross-turn too; a string/matrix field read back as scalar 0.
# A cross-turn IsStructArray element-field read now fetches via get_mat.
run_case "xturn_structarray_field_string" "$(cat <<'EOF'
s(1).Header = "name1";
disp(s(1).Header)
exit
EOF
)" "name1"

# 8j. #258 — a struct-array-RETURNING assignment (`s = fastaread(...)`) must
# persist via matlab_ws_set_struct_arr (kind=14) so a later turn's `s(i).Field`
# rehydrates the array + its elements.  Pre-fix it fell to set_mat (kind=1) and
# `s(1).Sequence` came back undef cross-turn.  Cleared bioinfo/fasta_align.
run_case "xturn_structarray_fn_return" "$(cat <<'EOF'
fastawrite("/tmp/reg_sa_xturn.fa", "recA", "ACGT");
s = fastaread("/tmp/reg_sa_xturn.fa");
disp(s(1).Sequence)
exit
EOF
)" "ACGT"

# 8k. #258 — a struct-array COPY `t = s` must propagate struct-array-ness so
# `t(i).Field` rehydrates cross-turn (and `t` persists as kind=14).  Pre-fix
# the copy lost the tag and `t(1).Header` fell to undef.
run_case "xturn_structarray_copy" "$(cat <<'EOF'
s(1).Header = "name1";
t = s;
disp(t(1).Header)
exit
EOF
)" "name1"

# 8i. #261 — a cross-turn class-pinned binding (dlarray weight) that is also
# REASSIGNED inside a loop turn lost its pin at an earlier read in that turn,
# because the cross-turn re-pin (applyWorkspaceKind) only ran on the auto-
# declare path; the reassignment set the pin only after the read.  So
# `logits = W2_dl * reshape(...)` didn't infer object<dlarray> and the
# activation (`softmax(logits)`) backed off to "unsupported call shape".  The
# Resolver now applies the cross-turn pin early for an assigned-in-TU binding.
# Absence-assertion: the softmax back-off must no longer appear (the loop body
# fails to compile pre-fix; the lenient REPL would still run a trailing disp).
run_case_absent "xturn_dlarray_reassign_in_loop" "$(cat <<'EOF'
W2_dl = dlarray(ones(3, 16));
for k = 1:2
    logits = W2_dl * reshape(dlarray(ones(2, 2, 4, 4)), 16, 4);
    yhat   = softmax(logits);
    W2_dl  = dlarray(ones(3, 16) - 0.2);
end
disp(99)
exit
EOF
)" "unsupported call shape for built-in function 'softmax'"

# 8h. #260 Symptom B — the REPL line accumulator counted the indexing `end` in
# `x(2:end)` as a block-`end`, so an if/for/while block containing one was
# submitted before its real `end`.  A classdef prelude then prepended to that
# still-open block produced `unexpected 'classdef' in expression`.  blockDepth
# now ignores an `end` inside ()/[]/{} (indexing), so the block accumulates to
# its true terminator and runs.  sum(x(2:end)) = 20+30 = 50.
run_case "repl_block_indexing_end_accum" "$(cat <<'EOF'
x = [10 20 30];
if true
  y = sum(x(2:end));
  disp(y);
end
exit
EOF
)" "50"

# 9. #240 — mpcmove with a p×ny reference-preview MATRIX on the REPL/JIT path.
# The mpc object round-trips through the workspace and loses its class pin, so
# the AOT `mpcmove -> matlab_mpc_move` rewrite (gated on that pin) is skipped.
# Without the LowerTensorOps fallback the call errors with "unsupported call
# shape" and the loop silently degrades (RMS ~1.9 m); with it, the REPL tracks
# like AOT (RMS ~0.014 m). Uses the real example as the canonical repro; the
# "RMS = 0.0" substring distinguishes the fixed result from the broken 1.8991.
QUAD_240="$(cd "$(dirname "$0")/../.." && pwd)/examples/quadrotor/quadrotor_pid_mpc.m"
run_case "mpcmove_preview_matrix_repl_240" "$(cat "$QUAD_240"; printf '\nexit\n')" "RMS = 0.0"

# Current-folder function files (#284). MATLAB makes every `.m` file in the
# working directory callable by name. Launch the REPL from a scratch dir
# holding add_two.m (function) + outer.m (which calls add_two), plus a plain
# script that must NOT become callable.
cfdir="$(mktemp -d)"
cat > "$cfdir/add_two.m" <<'EOF'
function y = add_two(x)
  y = x + 2;
end
EOF
cat > "$cfdir/outer.m" <<'EOF'
function y = outer(x)
  y = add_two(x) * 10;
end
EOF
cat > "$cfdir/plain_script.m" <<'EOF'
z = 99;
disp(z)
EOF
# Direct call into a current-folder function.
run_case_dir "$cfdir" "cwd_fn_direct" "$(cat <<'EOF'
disp(add_two(40))
exit
EOF
)" want "42"
# Transitive: a cwd function calling another cwd function.
run_case_dir "$cfdir" "cwd_fn_transitive" "$(cat <<'EOF'
disp(outer(4))
exit
EOF
)" want "60"
# A plain script file is not a function and must not be injected/run as one —
# calling it must not execute its body (no "99").
run_case_dir "$cfdir" "cwd_script_not_callable" "$(cat <<'EOF'
plain_script(1);
exit
EOF
)" absent "99"
rm -rf "$cfdir"

# cd / pwd (interpreter working directory). Build a parent dir with two child
# folders, each holding a function file, and drive the REPL from the parent:
# cd into a child, call its function, and read the cwd back with pwd. Exercises
# the call form cd('dir'), command forms `cd dir` / `cd ..`, and pwd.
cdroot="$(mktemp -d)"
mkdir -p "$cdroot/alpha" "$cdroot/beta"
cat > "$cdroot/alpha/ga.m" <<'EOF'
function y = ga(x)
  y = x + 1000;
end
EOF
cat > "$cdroot/beta/gb.m" <<'EOF'
function y = gb(x)
  y = x + 2000;
end
EOF
# cd('dir') call form, then call a function from that folder.
run_case_dir "$cdroot" "cd_call_form" "$(cat <<EOF
cd('$cdroot/alpha')
disp(ga(1))
exit
EOF
)" want "1001"
# cd dir command form + cd .. to switch folders mid-session.
run_case_dir "$cdroot" "cd_command_and_dotdot" "$(cat <<EOF
cd alpha
disp(ga(2))
cd ..
cd beta
disp(gb(2))
exit
EOF
)" want "2002"
# pwd reflects a prior cd.
run_case_dir "$cdroot" "pwd_reflects_cd" "$(cat <<EOF
cd('$cdroot/beta')
disp(pwd)
exit
EOF
)" want "$cdroot/beta"
# Dynamic cd(pathVar): a path held in a (cross-turn) variable, then call a
# function from the folder it selects.
run_case_dir "$cdroot" "cd_dynamic_var" "$(cat <<EOF
p = '$cdroot/alpha';
cd(p)
disp(ga(3))
exit
EOF
)" want "1003"
# pwd round-trip: remember the cwd, cd elsewhere, cd back via the variable.
run_case_dir "$cdroot/alpha" "cd_pwd_roundtrip" "$(cat <<EOF
home = pwd;
cd('$cdroot/beta')
disp(gb(3))
cd(home)
disp(ga(3))
exit
EOF
)" want "1003"
rm -rf "$cdroot"

# #265 — a char-array variable in a comparison / arithmetic operates on the
# character codes (like MATLAB), not as a string. `c == 'l'` used to leave
# matlab.eq unconverted (so the REPL printed nothing) and `c + 1` wrongly
# concatenated ("hello1"). The fix converts a char operand to a code matrix
# (matlab_string_to_codes) before the numeric op; genuine string concat is
# untouched.
# Matches the issue's exact repro (a single piped line). The char variable
# and its use share one compilation unit, so `c` keeps its char-array type.
run_case "char_var_compare_codes" "$(cat <<'EOF'
c='hello'; disp(c == 'l')
exit
EOF
)" "1         1         0"
run_case "char_var_add_codes" "$(cat <<'EOF'
c='hello'; disp(c + 1)
exit
EOF
)" "105       102       109       109       112"
# Genuine string concatenation must NOT regress.
run_case "string_var_concat_unaffected" "$(cat <<'EOF'
s = "hi"; disp(s + "!")
exit
EOF
)" "hi!"

# #289 — CROSS-TURN char arithmetic. The char variable is defined in one REPL
# input and used in a later one, so it round-trips through the workspace. It's
# stored under the dedicated char kind (matlab_ws_set_char, kind=18) and the
# Resolver stamps Binding::IsChar, so `c == 'l'` / `c + 1` still evaluate on
# character codes across turns. disp(c) must still print TEXT (not codes), and
# a genuine "..." string must still concatenate.
run_case "xturn_char_compare_codes" "$(cat <<'EOF'
c = 'hello';
disp(c == 'l')
exit
EOF
)" "1         1         0"
run_case "xturn_char_add_codes" "$(cat <<'EOF'
c = 'hello';
disp(c + 1)
exit
EOF
)" "105       102       109       109       112"
# disp of the char var across a turn still renders text, not a code matrix.
run_case "xturn_char_disp_text" "$(cat <<'EOF'
c = 'hello';
disp(c)
exit
EOF
)" "hello"
# A genuine string across a turn must still concatenate (not codes-convert).
run_case "xturn_string_concat_unaffected" "$(cat <<'EOF'
s = "hi";
disp(s + "!")
exit
EOF
)" "hi!"
# A char binding reassigned to a "..." string drops its char-ness.
run_case "xturn_char_then_string" "$(cat <<'EOF'
c = 'ab';
c = "xy";
disp(c + "z")
exit
EOF
)" "xyz"

# #291 — persistent REPL history. With MATLABC_HISTFILE set, each entered
# line is appended to the file, and a later session loads it back. A piped
# run with no env var must NOT create ~/.matlabc_history (no pollution).
histfile="$(mktemp)"
printf 'aa = 1;\nbb = 2;\nexit\n' | MATLABC_HISTFILE="$histfile" "$MATLABC" -repl >/dev/null 2>&1
printf 'cc = 3;\nexit\n'          | MATLABC_HISTFILE="$histfile" "$MATLABC" -repl >/dev/null 2>&1
hist="$(cat "$histfile" 2>/dev/null)"
if [[ "$hist" == $'aa = 1;\nbb = 2;\ncc = 3;' ]]; then
  pass=$((pass+1))
else
  fail=$((fail+1))
  fails+=("repl_persistent_history (history file content unexpected)")
  echo "FAIL  repl_persistent_history"
  echo "----- got -----"; echo "$hist"
fi
rm -f "$histfile"
# Pollution guard: a piped session without the env var leaves no default file.
# (Only meaningful if the dev box has no pre-existing ~/.matlabc_history.)
if [[ ! -e "$HOME/.matlabc_history" ]]; then
  printf 'dd = 4;\nexit\n' | "$MATLABC" -repl >/dev/null 2>&1
  if [[ -e "$HOME/.matlabc_history" ]]; then
    fail=$((fail+1))
    fails+=("repl_history_no_pollution (piped run created ~/.matlabc_history)")
    echo "FAIL  repl_history_no_pollution"
    rm -f "$HOME/.matlabc_history"
  else
    pass=$((pass+1))
  fi
fi

# #404 — exist(name [, kind]) status codes. Variable lookup uses the live
# REPL workspace; file/dir consult the filesystem. A private temp file/dir
# keeps the assertions deterministic.
exist_dir="$(mktemp -d)"
exist_file="$exist_dir/probe.txt"
printf 'x\n' > "$exist_file"
run_case "exist_var" "$(cat <<EOF
myvar = 42;
fprintf('EXIST_VAR=%d\n', exist('myvar','var'));
exit
EOF
)" "EXIST_VAR=1"
run_case "exist_var_missing" "$(cat <<EOF
fprintf('EXIST_NOVAR=%d\n', exist('no_such_var','var'));
exit
EOF
)" "EXIST_NOVAR=0"
run_case "exist_file" "$(cat <<EOF
fprintf('EXIST_FILE=%d\n', exist('$exist_file','file'));
exit
EOF
)" "EXIST_FILE=2"
run_case "exist_file_missing" "$(cat <<EOF
fprintf('EXIST_NOFILE=%d\n', exist('$exist_dir/nope.txt','file'));
exit
EOF
)" "EXIST_NOFILE=0"
run_case "exist_dir" "$(cat <<EOF
fprintf('EXIST_DIR=%d\n', exist('$exist_dir','dir'));
exit
EOF
)" "EXIST_DIR=7"
run_case "exist_no_kind_var" "$(cat <<EOF
zz = 1;
fprintf('EXIST_NOKIND=%d\n', exist('zz'));
exit
EOF
)" "EXIST_NOKIND=1"
rm -rf "$exist_dir"

# #405 — error()/try-catch: identifier + printf-formatted message on the
# caught MException, plain-message form, and error() halting its function.
run_case "err_id_and_message" "$(cat <<'EOF'
try
  error('myPkg:bad', 'value was %d', 42);
catch ME
  fprintf('EID=%s\n', ME.identifier);
end
exit
EOF
)" "EID=myPkg:bad"
run_case "err_message_formatted" "$(cat <<'EOF'
try
  error('myPkg:bad', 'value was %d', 42);
catch ME
  fprintf('EMSG=%s\n', ME.message);
end
exit
EOF
)" "EMSG=value was 42"
run_case "err_plain_message" "$(cat <<'EOF'
try
  error('simple boom');
catch ME
  fprintf('PID=[%s] PMSG=%s\n', ME.identifier, ME.message);
end
exit
EOF
)" "PID=[] PMSG=simple boom"
# error() halts the enclosing function: the line after error() must not run.
run_case "err_halts_before" "$(cat <<'EOF'
function r = ff()
  fprintf('F_BEFORE\n');
  error('stop');
  fprintf('F_AFTER\n');
  r = 1;
end
ff()
exit
EOF
)" "F_BEFORE"
run_case_absent "err_halts_skips_after" "$(cat <<'EOF'
function r = ff()
  fprintf('F_BEFORE\n');
  error('stop');
  fprintf('F_AFTER\n');
  r = 1;
end
ff()
exit
EOF
)" "F_AFTER"

# #423 — interpreter raises MATLAB-style errors instead of silently returning
# 0/empty: undefined names, and out-of-bounds / invalid indexing.
run_case "undef_var_read" "$(printf 'nope_var\nexit\n')" \
  "Unrecognized function or variable 'nope_var'."
run_case "undef_fn_call" "$(printf 'nope_fn(3)\nexit\n')" \
  "Unrecognized function or variable 'nope_fn'."
run_case "undef_in_expr" "$(printf 'y = missing_thing + 1\nexit\n')" \
  "Unrecognized function or variable 'missing_thing'."
run_case "index_oob" "$(printf 'a = [1 2 3];\na(5)\nexit\n')" \
  "Index exceeds the number of array elements. Index must not exceed 3."
run_case "index_zero" "$(printf 'a = [1 2 3];\na(0)\nexit\n')" \
  "Array indices must be positive integers or logical values."
run_case "index_frac" "$(printf 'a = [1 2 3];\na(1.5)\nexit\n')" \
  "Array indices must be positive integers or logical values."
run_case "index_2d_oob" "$(printf 'm = [1 2; 3 4];\nm(3,1)\nexit\n')" \
  "Index in position 1 exceeds array bounds. Index must not exceed 2."
run_case "cell_oob" "$(printf 'c = {1, 2};\nc{5}\nexit\n')" \
  "Index exceeds the number of elements in the cell array. Index must not exceed 2."
# Cell PAREN-index (c(i) returns a 1x1 sub-cell) is bounds-checked too — it
# used to mis-route to the matrix subscript and yield a garbage-bound value.
# Single-turn (one input) so `c` is a cell binding in the same TU; cross-turn
# REPL re-tagging of a cell from the workspace is a separate gap (see #431).
run_case "cell_paren_oob" "$(printf 'c = {1, 2}; c(3)\nexit\n')" \
  "Index exceeds the number of elements in the cell array. Index must not exceed 2."
run_case "cell_paren_zero" "$(printf 'c = {1, 2}; c(0)\nexit\n')" \
  "Array indices must be positive integers or logical values."
run_case "cell_paren_ok" "$(printf 'c = {10, 20}; d = c(1); fprintf("P=%%d\\n", d{1});\nexit\n')" \
  "P=10"
# Valid indexing and a defined var must still work (no over-eager raising).
run_case "index_valid_still_works" "$(printf 'a = [10 20 30];\nfprintf("V=%%d\\n", a(2));\nexit\n')" \
  "V=20"
# Bounds checks also cover complex and N-D reads (not just real 2-D).
run_case "index_complex_oob" "$(printf 'z = complex([1 2 3], [4 5 6]);\nz(5)\nexit\n')" \
  "Index exceeds the number of array elements. Index must not exceed 3."
run_case "index_3d_linear_oob" "$(printf 'a = zeros(2,2,2);\na(9)\nexit\n')" \
  "Index exceeds the number of array elements. Index must not exceed 8."
run_case "index_3d_pos_oob" "$(printf 'a = zeros(2,2,2);\na(3,1,1)\nexit\n')" \
  "Index in position 1 exceeds array bounds. Index must not exceed 2."
run_case "index_nd_valid_works" "$(printf 'a = reshape(1:8, 2, 2, 2);\nfprintf("W=%%d\\n", a(8));\nexit\n')" \
  "W=8"

# Element-wise binary ops on incompatible sizes raise instead of silently
# reading past the shorter operand and returning garbage.
run_case "ewise_add_mismatch" "$(printf '[1 2 3] + [1 2]\nexit\n')" \
  "Arrays have incompatible sizes for this operation."
run_case "ewise_cmp_mismatch" "$(printf '[1 2 3] == [1 2]\nexit\n')" \
  "Arrays have incompatible sizes for this operation."
run_case "ewise_pow_mismatch" "$(printf '[1 2 3] .^ [1 2]\nexit\n')" \
  "Arrays have incompatible sizes for this operation."
run_case "ewise_mod_mismatch" "$(printf 'mod([1 2 3], [1 2])\nexit\n')" \
  "Arrays have incompatible sizes for this operation."
run_case "ewise_max_mismatch" "$(printf 'max([1 2 3], [1 2])\nexit\n')" \
  "Arrays have incompatible sizes for this operation."
# Equal-size and scalar-broadcast element-wise ops still work.
run_case "ewise_equal_size_ok" "$(printf 'fprintf("S=%%d\\n", sum([1 2 3] + [4 5 6]));\nexit\n')" \
  "S=21"
run_case "ewise_scalar_ok" "$(printf 'fprintf("S2=%%d\\n", sum([1 2 3] + 10));\nexit\n')" \
  "S2=36"

# Indexed-assignment size mismatch errors instead of silently dropping/mis-
# assigning; scalar broadcast and matched sizes still work.
run_case "assign_size_mismatch" "$(printf 'x=[1 2 3]; x(1:2)=[5 6 7]\nexit\n')" \
  "Unable to perform assignment because the size of the left side is 1-by-2 and the size of the right side is 1-by-3."
run_case "assign_mask_mismatch" "$(printf 'v=[1 2 3 4]; v(v>2)=[10 20 30]\nexit\n')" \
  "Unable to perform assignment because the size of the left side is 2-by-1 and the size of the right side is 1-by-3."
run_case "assign_matched_ok" "$(printf 'x=[1 2 3]; x(1:2)=[5 6]; fprintf("A=%%d\\n", sum(x));\nexit\n')" \
  "A=14"
run_case "assign_scalar_bcast_ok" "$(printf 'x=[1 2 3]; x(1:2)=9; fprintf("B=%%d\\n", sum(x));\nexit\n')" \
  "B=21"

# Concatenation dimension mismatches error instead of returning a silent
# empty; valid shapes still work. (matmul-mismatch is deferred — it exposes a
# separate mpcmove 0x0 bug; see the tracking issue.)
run_case "vertcat_mismatch" "$(printf '[ [1 2 3]; [1 2] ]\nexit\n')" \
  "Dimensions of arrays being concatenated are not consistent."
run_case "horzcat_mismatch" "$(printf '[ [1;2], [3;4;5] ]\nexit\n')" \
  "Dimensions of arrays being concatenated are not consistent."
run_case "concat_ok" "$(printf 'fprintf("C=%%d\\n", sum([[1 2],[3 4]]));\nexit\n')" \
  "C=10"

# #431 item 1: reading a field that does not exist on a struct raises
# "Unrecognized field name" instead of silently returning 0; present fields
# still read fine.
run_case "struct_unknown_field" "$(printf 's.a = 1; s.b\nexit\n')" \
  "Unrecognized field name \"b\"."
run_case "struct_known_field_ok" "$(printf 's.a = 1; s.b = 2; fprintf("F=%%d\\n", s.a + s.b);\nexit\n')" \
  "F=3"

# #433: reshape count mismatch, non-square inv/det, too-many-outputs, and
# anonymous-function arity all raise MATLAB-style errors instead of silently
# returning wrong/empty values; valid forms still work.
run_case "reshape_count_mismatch" "$(printf 'reshape([1 2 3 4],2,3)\nexit\n')" \
  "Number of elements must not change."
run_case "reshape_ok" "$(printf 'B=reshape([1 2 3 4 5 6],2,3); fprintf("R=%%d\\n", sum(B(:)));\nexit\n')" \
  "R=21"
run_case "inv_nonsquare" "$(printf 'inv([1 2 3])\nexit\n')" \
  "Matrix must be square."
run_case "det_nonsquare" "$(printf 'det([1 2 3])\nexit\n')" \
  "Matrix must be square."
run_case "det_square_ok" "$(printf 'fprintf("D=%%d\\n", det([1 2;3 4]));\nexit\n')" \
  "D=-2"
run_case "too_many_outputs" "$(printf '[a,b] = sin(1)\nexit\n')" \
  "Too many output arguments."
run_case "anon_too_few_args" "$(printf 'f = @(a,b) a + b; f(1)\nexit\n')" \
  "Not enough input arguments."
run_case "anon_too_many_args" "$(printf 'g = @(a) a; g(1,2)\nexit\n')" \
  "Too many input arguments."
run_case "anon_arity_ok" "$(printf 'f = @(a,b) a + b; fprintf("S=%%d\\n", f(3,4));\nexit\n')" \
  "S=7"

# #433: mldivide (\) row mismatch raises "Matrix dimensions must agree."
# (rows-agree underdetermined / non-square solves are a separate gap, untouched).
run_case "mldivide_row_mismatch" "$(printf '[1 2; 3 4] \\ [1 2 3]\nexit\n')" \
  "Matrix dimensions must agree."
run_case "mldivide_ok" "$(printf 'x = [1 2;3 4] \\ [5;6]; fprintf("X=%%.1f\\\\n", sum(x));\nexit\n')" \
  "X=0.5"

echo "----"
echo "passed: $pass    failed: $fail"
if (( fail > 0 )); then
  echo "failures:"
  for f in "${fails[@]}"; do echo "  - $f"; done
fi
exit $(( fail > 0 ? 1 : 0 ))
