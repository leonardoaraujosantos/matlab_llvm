% regress_fprintf_reduction.m — regression test for fprintf with a
% data argument that is a reduction result (max / sum / etc.).  Sema
% types those as `any`, so at runtime they arrive as a 1x1 matrix
% pointer rather than an f64.  Before the fix, the fprintf lowering
% bailed when a data operand was not exactly f64, which left the
% format-string matlab.const_char op unlowered and broke translation.
% The lowering now extracts element 1 from a ptr operand via
% matlab_subscript1_s.

a = [3, 1, 2];

% --- reduction result passed straight to fprintf -----------------
fprintf('max = %.1f\n', max(a));
fprintf('sum = %.1f\n', sum(a));
fprintf('min = %.1f\n', min(a));
fprintf('mean = %.2f\n', mean(a));

% --- reduction stored first, then printed ------------------------
m = max(a);
fprintf('stored max = %.1f\n', m);

% --- plain f64 args still work ------------------------------------
fprintf('plain %.2f and %.0f\n', 3.14159, 7);

% --- format string with no data args still works -----------------
fprintf('no args here\n');

% --- two reduction results in one call ---------------------------
fprintf('span %.1f to %.1f\n', min(a), max(a));
