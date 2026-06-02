% regress_matrix_if_cond.m — #120: a matrix-valued comparison used directly
% as an if / elseif condition (`if abs(v) < tol`) produced a tensor<*xi1>
% that scf.if rejected ("operand #0 must be 1-bit signless integer, but got
% tensor<*xi1>") — the matrix condition was never reduced to MATLAB's
% "true iff every element is true".  fixupIfCond now wraps a tensor condition
% in matlab_mat_truth, the same reduction the matrix-pointer path already used.

v = [0.001 0.002 0.003];
% every element < 1e-2  ->  condition true
if abs(v) < 1e-2; disp(1); else; disp(0); end

w = [0.001 5.0 0.003];
% mixed  ->  not all true  ->  condition false
if abs(w) < 1e-2; disp(0); else; disp(1); end

% matrix condition in an elseif arm too
if abs(w) < 1e-4; disp(0); elseif abs(w) < 1e-2; disp(0); else; disp(1); end

% plain relational matrix condition
if v > 0; disp(1); else; disp(0); end

% MATLAB truthiness of a vector with a zero element: `if [1 0 1]` is false
z = [1 0 1];
if z; disp(0); else; disp(1); end
