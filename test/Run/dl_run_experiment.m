% dl_run_experiment — T6.8 gating: programmatic experiment-sweep harness.
%
% Cartesian sweep over a 3 x 3 grid (learning_rate x hidden_size) using
% runExperiment(@trialFn, Grid).  Each row of Grid is one trial; trialFn
% receives a 2x1 column [lr; h] and returns a scalar surrogate loss.
%
% NB: closure captures into the objective handle aren't supported (same
% trap as bayesopt), so the surrogate uses inline constants.

% Build the 9x2 Cartesian grid: lr in {0.01, 0.1, 0.5}, h in {4, 8, 16}.
Grid = zeros(9, 2);
Grid(1, 1) = 0.01; Grid(1, 2) = 4.0;
Grid(2, 1) = 0.01; Grid(2, 2) = 8.0;
Grid(3, 1) = 0.01; Grid(3, 2) = 16.0;
Grid(4, 1) = 0.10; Grid(4, 2) = 4.0;
Grid(5, 1) = 0.10; Grid(5, 2) = 8.0;
Grid(6, 1) = 0.10; Grid(6, 2) = 16.0;
Grid(7, 1) = 0.50; Grid(7, 2) = 4.0;
Grid(8, 1) = 0.50; Grid(8, 2) = 8.0;
Grid(9, 1) = 0.50; Grid(9, 2) = 16.0;

% Surrogate trial: bowl around (lr=0.10, h=8.0).  Inline constants only.
trial = @(p) (p(1) - 0.10) * (p(1) - 0.10) * 100.0 + ...
             (p(2) - 8.0)  * (p(2) - 8.0)  * 0.05;

results_raw = runExperiment(trial, Grid);

% Materialize results into a known-shape column to enable variable-index
% subscript (none-typed ptrs can't subscript with a loop var).
results = zeros(9, 1);
for k = 1:9
    results(k, 1) = results_raw(k);
end

% Find best (minimum-loss) trial.
best_loss = results(1, 1);
best_idx  = 1;
for k = 2:9
    if results(k, 1) < best_loss
        best_loss = results(k, 1);
        best_idx  = k;
    end
end

best_lr = Grid(best_idx, 1);
best_h  = Grid(best_idx, 2);

fprintf('dl_run_experiment: best trial = %d (lr=%.2f, h=%.0f)\n', ...
        best_idx, best_lr, best_h);
fprintf('dl_run_experiment: best loss = %.4f\n', best_loss);

ok = (abs(best_lr - 0.10) < 1e-6) && (abs(best_h - 8.0) < 1e-6);
if ok
    fprintf('dl_run_experiment: PASS\n');
else
    fprintf('dl_run_experiment: FAIL\n');
end
