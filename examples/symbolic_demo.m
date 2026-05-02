% Symbolic Math Toolbox demo — exercises the matlab_llvm/SymPP integration
% across the major function groups. Build with -DMATLAB_LLVM_WITH_SYM=ON
% and run through -emit-cpp / -emit-llvm or the JIT REPL.
%
% Surface covered:
%   declaration      : syms, sym, str2sym
%   arithmetic       : + - * / ^ on sym (pure + mixed-mode with double)
%   calculus         : diff (1st + nth), int (indef + def), taylor, limit
%   algebra          : simplify, expand, factor, subs
%   single-eq solve  : solve(eq, x), solve(f == 0, x)
%   numeric eval     : double(s), vpa(s, dps), vpasolve, nsolve
%   ODE / PDE        : dsolve (1st + 2nd order), dsolve_ivp, checkodesol,
%                      pdsolve_heat, pdsolve_wave
%   transforms       : laplace + ilaplace, fourier + ifourier, ztrans + iztrans
%   assumptions      : assume, assumeAlso, clearAssumptions
%   matrices         : sym_matrix, sym_eye, sym_det, sym_inv, sym_linsolve,
%                      sym_dsolve_system, sym_solve_2x2

% --- Declaration & basic arithmetic -------------------------------------
syms x
syms a b c

f = a*x^2 + b*x + c;
disp('f = a*x^2 + b*x + c:');
disp(f)

% Arithmetic dispatch: + - * / ^ on sym, mixed-mode with f64 literals.
g = (x + 1)*(x - 1) + 2*x^3 - x/2;
disp('g = (x+1)(x-1) + 2*x^3 - x/2:');
disp(g)

% --- Calculus -----------------------------------------------------------
disp('diff(f, x) — first derivative wrt x:');
disp(diff(f, x))

disp('diff(f, x, 2) — second derivative:');
disp(diff(f, x, 2))

disp('int(x^3, x) — indefinite integral:');
disp(int(x^3, x))

disp('int(x^2, x, sym(0), sym(1)) — definite integral on [0, 1]:');
disp(int(x^2, x, sym(0), sym(1)))

disp('taylor(sin(x), x, 0, 5) — degree-5 Taylor about x = 0:');
disp(taylor(sin(x), x, sym(0), 5))

disp('limit(sin(x)/x, x, 0) — classic 1:');
disp(limit(sin(x)/x, x, sym(0)))

% --- Algebra ------------------------------------------------------------
disp('simplify(sin(x)^2 + cos(x)^2) — expect 1:');
disp(simplify(sin(x)^2 + cos(x)^2))

disp('expand((x + 1)^4):');
disp(expand((x + 1)^4))

disp('factor(x^2 - 1, x):');
disp(factor(x^2 - 1, x))

disp('subs(x + 1, x, 2) — substitute x = 2:');
disp(subs(x + 1, x, sym(2)))

% --- Single-equation solvers --------------------------------------------
disp('solve(x^2 - 5*x + 6, x) — quadratic roots:');
disp(solve(x^2 - 5*x + 6, x))

disp('solve(x^2 == 4, x) — relational form:');
disp(solve(x^2 == 4, x))

% Newton's method numeric solve (at variable precision via MPFR).
disp('vpasolve(cos(x) - x, x, sym(1), 32) — Dottie number to 32 digits:');
disp(vpasolve(cos(x) - x, x, sym(1), 32))

% --- Numeric evaluation -------------------------------------------------
disp('double(subs(x + 1, x, 2)) — sym -> f64:');
disp(double(subs(x + 1, x, sym(2))))

disp('vpa(sym(pi), 32) — pi to 32 decimal digits:');
disp(vpa(sym(pi), 32))

% --- Assumptions --------------------------------------------------------
% Assumptions register in SymPP's side-table and rebind the variable to
% a fresh symbol carrying the mask. Currently `simplify` is structural
% and does NOT auto-honour assumptions — use refine() in SymPP for that.
syms p
assume(p, 'positive')
assumeAlso(p, 'real')
disp('assumptions on p (after assume positive + real) — sym(p) carries the mask');
disp(p)
clearAssumptions(p)

% --- Ordinary differential equations -----------------------------------
% SymPP's dsolve takes the unknown function and its derivatives as plain
% symbols (no AppliedFunction y(x) shape). For y' + y = 0:
syms y yp
disp('dsolve(yp + y, y, yp, x) — first-order linear ODE y'' + y = 0:');
disp(dsolve(yp + y, y, yp, x))

% Second-order: y'' + y = 0 -> auto-classified as constant-coefficient.
syms ypp
disp('dsolve(ypp + y, y, yp, ypp, x) — second-order y'''' + y = 0:');
disp(dsolve(ypp + y, y, yp, ypp, x))

% Apply an initial condition: y' + y = 0, y(0) = 1 -> exp(-x).
disp('dsolve_ivp(yp + y, y, yp, x, 0, 1) — IVP, expect exp(-x):');
disp(dsolve_ivp(yp + y, y, yp, x, sym(0), sym(1)))

% Verify the solution.
sol = dsolve_ivp(yp + y, y, yp, x, sym(0), sym(1));
disp('checkodesol(yp + y, sol, y, yp, x) — residual, expect 0:');
disp(checkodesol(yp + y, sol, y, yp, x))

% --- Partial differential equations ------------------------------------
% Heat equation u_t = k * u_xx via separation of variables.
syms k lam t
disp('pdsolve_heat(k, lambda, x, t) — separation-of-variables form:');
disp(pdsolve_heat(k, lam, x, t))

% Wave equation u_tt = c^2 * u_xx via d''Alembert.
syms wc
disp('pdsolve_wave(c, x, t) — d''Alembert F(x - c*t) + G(x + c*t):');
disp(pdsolve_wave(wc, x, t))

% --- Integral transforms -----------------------------------------------
syms s w n z
disp('laplace(exp(-a*t), t, s) — expect 1/(s + a):');
disp(laplace(exp(-a*t), t, s))

disp('fourier(exp(-t*t), t, w) — Gaussian, expect sqrt(pi)*exp(-w^2/4):');
disp(fourier(exp(-t*t), t, w))

disp('ztrans(sym(1), n, z) — DC unit, expect z/(z - 1):');
disp(ztrans(sym(1), n, z))

% --- Symbolic matrices --------------------------------------------------
% sym_matrix(R, C, e11, e12, ..., eRC) takes integer-literal R, C followed
% by R*C scalar sym entries (row-major). The standard `[a 1; 2 b]` matrix
% literal syntax doesn't yet detect sym entries — that's Phase 6.2.
M = sym_matrix(2, 2, a, sym(1), sym(2), b);
disp('M = [[a, 1], [2, b]] — symbolic 2x2 matrix:');
disp(M)

disp('sym_det(M) — symbolic determinant:');
disp(sym_det(M))

disp('sym_inv(M) — symbolic inverse:');
disp(sym_inv(M))

% Solve A * x = b symbolically.
A = sym_matrix(2, 2, sym(1), sym(2), sym(3), sym(4));
bv = sym_matrix(2, 1, sym(1), sym(2));
xs = sym_linsolve(A, bv);
disp('sym_linsolve([[1,2],[3,4]], [1; 2]) — expect [0; 1/2]:');
disp(xs)

% Linear ODE system y'' = A * y returns a column-vector general solution.
syms tx
A2 = sym_matrix(2, 2, sym(0), sym(1), sym(-1), sym(0));
disp('sym_dsolve_system([[0,1],[-1,0]], tx) — y'' = A*y for the rotation matrix:');
disp(sym_dsolve_system(A2, tx))

% --- Multi-equation solver ---------------------------------------------
% sym_solve_2x2 returns a symmat where each row is a joint solution.
syms u v
disp('sym_solve_2x2(u^2 + v^2 - 1, v - u, u, v) — circle ∩ y=x:');
disp(sym_solve_2x2(u^2 + v^2 - 1, v - u, u, v))
