% Regression: NaN / Inf / -Inf literals must emit valid code on every backend.
% emit-python emitted bare `nan`/`inf` (NameError) and emit-typescript bare
% `nan`/`inf` (ReferenceError); AOT was fine. Checked via comparisons so the
% test is independent of NaN/Inf display formatting. (#197)
n = NaN;
if n == n; disp(1); else; disp(0); end       % NaN != NaN -> 0
p = Inf;
if p > 1e300; disp(1); else; disp(0); end     % Inf -> 1
m = -Inf;
if m < -1e300; disp(1); else; disp(0); end    % -Inf -> 1
v = [1 NaN 3];                                 % NaN in a matrix literal
e = v(2);
if e == e; disp(1); else; disp(0); end         % NaN element -> 0
fprintf('sum_finite=%.0f\n', 1 + 2 + 3);        % sanity: normal floats unaffected
