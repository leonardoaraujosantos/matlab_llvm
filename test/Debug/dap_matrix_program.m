% Fixture for matrix-expansion in the LOCALS / Watch panel. The
% breakpoint at the last line gives the script frame two matrices of
% different shapes (a 2x3 and a 3-element column vector) plus a 1x1
% scalar-shaped matrix so the test can assert all three formatting
% paths (RxC label, column vector, scalar unbox). Line numbers are
% referenced by dap_scenarios.py.
A = [1 2 3; 4 5 6];
b = [10; 20; 30];
s = ones(1, 1) * 7;
disp(s);
