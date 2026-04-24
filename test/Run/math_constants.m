% pi / e / Inf / NaN / eps as numeric constants.
% `pi` and friends used to resolve as undefined names; they now
% fold to arith.constant at MLIR emit time.
fprintf('%.4f\n', pi);
fprintf('%.4f\n', e);
fprintf('%.0f\n', sin(pi));
fprintf('%.0f\n', cos(0));
fprintf('%.4f\n', 2 * pi);
fprintf('%.2f\n', eps * 1e16);
