% Tier-1 builtins demo: filter, any, all, tril, triu, fftshift, ifftshift,
% std, var, median, diff, meshgrid, ndgrid.
%
% Each call exercises both the runtime function and the multi-return
% splitter (for meshgrid / ndgrid). Outputs are hand-verified.

% --- filter --------------------------------------------------------------
% 4-tap moving average via FIR. Step input -> ramps from 0.25 up to 1.0
% and stays there. The denominator is [1 0] (i.e. 1 + 0*z^-1 == 1) — a
% multi-element vector since scalar singletons currently collapse to f64.
b = [0.25 0.25 0.25 0.25];
a = [1 0];
x = [1 1 1 1 1 1];
disp('filter([0.25 0.25 0.25 0.25], [1 0], ones(1,6)) — moving average:');
disp(filter(b, a, x));

% Single-pole IIR (first-order low-pass): y[n] = 0.5*x[n] + 0.5*y[n-1]
% with a unit-impulse input. Output is the geometric sequence
% 0.5, 0.25, 0.125, ...
disp('filter([0.5 0], [1 -0.5], [1 0 0 0 0]) — IIR impulse response:');
disp(filter([0.5 0], [1 -0.5], [1 0 0 0 0]));

% --- any / all -----------------------------------------------------------
v = [0 0 3 0 5];
disp('any([0 0 3 0 5]) (expect 1):');
disp(any(v));
disp('all([0 0 3 0 5]) (expect 0):');
disp(all(v));
disp('all([1 2 3 4 5]) (expect 1):');
disp(all([1 2 3 4 5]));

% Column-wise on a matrix.
M = [0 1 0;
     0 1 1;
     0 0 1];
disp('any(M) by column (expect [0 1 1]):');
disp(any(M));
disp('all(M) by column (expect [0 0 0]):');
disp(all(M));

% --- tril / triu ---------------------------------------------------------
A = [1 2 3;
     4 5 6;
     7 8 9];
disp('tril(A):');
disp(tril(A));
disp('triu(A):');
disp(triu(A));

% --- fftshift / ifftshift ------------------------------------------------
% On a length-4 vector [0 1 2 3] fftshift -> [2 3 0 1].
disp('fftshift([0 1 2 3]) (expect 2 3 0 1, complex layout):');
disp(fftshift([0 1 2 3]));
disp('ifftshift(fftshift([0 1 2 3])) round-trip:');
disp(ifftshift(fftshift([0 1 2 3])));

% --- std / var / median --------------------------------------------------
% var has the N-1 normalization, so var([1 2 3 4 5]) = 10/4 = 2.5.
v = [1 2 3 4 5];
disp('mean / std / var of 1:5:');
disp(mean(v));
disp(std(v));
disp(var(v));
disp('median([7 3 1 9 5]) (expect 5):');
disp(median([7 3 1 9 5]));
disp('median([1 2 3 4]) (expect 2.5):');
disp(median([1 2 3 4]));

% Column-wise on a matrix.
disp('std(A) by column:');
disp(std(A));

% --- diff ----------------------------------------------------------------
disp('diff([1 4 9 16 25]) — first differences:');
disp(diff([1 4 9 16 25]));
disp('diff(A) — column-wise differences:');
disp(diff(A));

% --- meshgrid / ndgrid ---------------------------------------------------
[X, Y] = meshgrid([10 20 30], [1 2]);
disp('[X, Y] = meshgrid([10 20 30], [1 2]):');
disp('X =');
disp(X);
disp('Y =');
disp(Y);

% Same data through ndgrid (transpose convention).
[Xn, Yn] = ndgrid([10 20 30], [1 2]);
disp('[Xn, Yn] = ndgrid([10 20 30], [1 2]):');
disp('Xn =');
disp(Xn);
disp('Yn =');
disp(Yn);
