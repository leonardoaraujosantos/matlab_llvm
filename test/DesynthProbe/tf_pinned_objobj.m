% #191 P3 — dispatch-desynth coverage contract, verified via the
% MATLAB_LLVM_PROBE_LATE_MONO probe (see test/DesynthProbe/run_tests.sh).
%
% Every operator-on-operands of a MIGRATED class (tf/ss/zpk/pid/frd/Vec2)
% where the object is on the LHS must be rewritten into an explicit method
% call at Sema time (the desynth pass) rather than left to the lowering
% synthesis fallback. That includes operands whose class identity is carried
% only by the binding's PinnedClass — e.g. a builtin like `c2d` returns a
% value typed `any` but pinned `tf` — which the original pass (keying solely
% off an object Expr->Ty) missed, emitting `tf__mtimes` from lowering instead.
%
% The scalar-on-the-LHS form (`k * G`) is now also desynthed, into
% `(tf(k)).mtimes(G)` (the lowering recovers the class from the ctor-call
% base's object type). So this fixture emits ZERO probe fires — full
% migration. Liveness for the probe wiring lives in `liveness_unmigrated.m`.

G = tf([1 2], [1 3 5]);
H = tf([1 1], [1 1]);

% obj-op-obj, both operands object-typed.
P = G * H;
disp(P.Numerator);

% obj-op-obj where both operands are PinnedClass=tf but Expr->Ty=any
% (c2d's result is not object-typed) — the case the fix newly covers.
Cz = c2d(G, 0.05, 'zoh');
Q = Cz * Cz;
disp(Q.Numerator);

% chained obj-op-obj mixing an object-typed call result with a pinned-only var.
R = (G * H) + Cz;
disp(R.Numerator);

% scalar-on-LHS: now desynthed to (tf(5)).mtimes(G).
B = 5 * G;
disp(B.Numerator);
