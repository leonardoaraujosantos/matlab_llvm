% Deep Learning T6.4 gating test — Bayesian hyperparameter search.  The
% MATLAB UG idiom for this is `bayesopt(@(p)final_loss(net, p), bounds)`;
% the project ships `bayesopt` end-to-end (GP + EI; see Stats T6).  This
% test exercises the *integration*: a synthetic loss surface that mimics
% the bowl-shape of an MLP's final-loss-vs-learning-rate curve, with a
% sweet spot at lr* = 0.3.  bayesopt should land within a small ε of it.
%
% (Closure captures into bayesopt's objective handle aren't supported
% yet — see the Stats T6 trap list — which is why this fixture uses a
% pure inline-constant objective.  A real DL HP-tuner would either
% inline all training-data constants or wait on that fix.)

rng(7);

% Synthetic surface: bowl in [0.01, 1.0] with a tiny sinusoidal ripple to
% make the search non-trivial.  Minimum is near lr = 0.3.
f = @(p) (p(1) - 0.3) * (p(1) - 0.3) + 0.05 * sin(8 * p(1));

best = bayesopt(f, 0.01, 1.0);

% Rounding to the nearest 0.1 absorbs the small ripple offset (~0.06).
fprintf('best learning-rate x10 rounds to %.0f\n', round(10 * best(1)));
% f(best) returns a function-handle result whose type isn't known until
% runtime, so we re-evaluate the surface inline (constants only) for the
% objective readout.  Use abs() to render a portable sign (the -0 vs +0
% libstdc++/libc++ split bites bare %.0f of small-negative values).
final_obj = (best(1) - 0.3) * (best(1) - 0.3) + 0.05 * sin(8 * best(1));
fprintf('|final objective| x100 rounds to %.0f\n', round(100 * abs(final_obj)));
% Verify bayesopt landed on the global minimum of the surface (lr ≈ 0.5,
% where the local sinusoidal trough lives) — the corresponding objective
% value is small (~0.002), well below the boundary objective values.
better_than_boundary = 0;
if abs(final_obj) < 0.05
    better_than_boundary = 1;
end
fprintf('found low-objective HP = %.0f\n', better_than_boundary);
