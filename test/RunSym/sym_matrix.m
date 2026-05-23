% Symbolic matrices: det / trace / transpose / inv / rank.
syms a b
M = [a, sym(1); sym(0), b];
disp(sym_det(M))           % a*b
disp(sym_trace(M))         % a + b
disp(sym_transpose(M))
N = [sym(2), sym(0); sym(0), sym(3)];
disp(sym_inv(N))
disp(sym_rank(N))          % 2
