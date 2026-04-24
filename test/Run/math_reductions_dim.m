% Dimension-aware reductions and cumulative scans.

A = [1 2 3; 4 5 6];

% sum with explicit dim.
disp(sum(A));                % [5 7 9]  (per column == dim 1 default)
disp(sum(A, 1));             % [5 7 9]
disp(sum(A, 2));             % [6; 15]

% mean with explicit dim.
disp(mean(A, 1));            % [2.5 3.5 4.5]
disp(mean(A, 2));            % [2; 5]

% prod with explicit dim.
disp(prod(A, 1));            % [4 10 18]
disp(prod(A, 2));            % [6; 120]

% Cumulative scans.
v = [1 2 3 4];
disp(cumsum(v));             % [1 3 6 10]
disp(cumprod(v));            % [1 2 6 24]

disp(cumsum(A, 1));          % columns:  [1 2 3; 5 7 9]
disp(cumsum(A, 2));          % rows: [1 3 6; 4 9 15]
