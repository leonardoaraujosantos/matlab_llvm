% Fixture for the DAP complex + 3-D matrix expansion scenarios.
% By line 5, A is a 2x2x2 real 3-D array and c is a 1x1 complex
% scalar (the simplest complex shape we can construct without
% pulling in matrix-level complex arithmetic). Line 5 is the
% breakpoint target.
A = ones(2, 2, 2);
A(1, 2, 1) = 42;
c = 3 + 4i;
disp(c);
