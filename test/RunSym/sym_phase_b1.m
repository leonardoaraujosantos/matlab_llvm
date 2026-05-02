syms a b
M = sym_matrix(2, 2, a, sym(1), sym(2), b);
disp(M)
disp(sym_det(M))
syms x
A = sym_matrix(2, 2, sym(1), sym(2), sym(3), sym(4));
bv = sym_matrix(2, 1, sym(1), sym(2));
xs = sym_linsolve(A, bv);
disp(xs)
disp(vpasolve(cos(x) - x, x, sym(1), 32))
syms y yp
sol = dsolve(yp + y, y, yp, x);
disp(sol)
disp(checkodesol(yp + y, sol, y, yp, x))
disp(dsolve_ivp(yp + y, y, yp, x, sym(0), sym(1)))
