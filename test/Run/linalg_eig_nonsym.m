% Tier 1.1 (Control System Toolbox roadmap §2.1) — non-symmetric eig
% via Francis double-shift QR on Hessenberg form. Symmetric matrices
% still take the existing Jacobi path (faster, simpler, real-typed
% return); non-symmetric A drops into the new path.
%
% Without this fix, eig(A) for a non-self-adjoint A would return the
% eigenvalues of (A+A')/2 — wrong for any control plant.

% --- 1. Triangular (lower) — eigenvalues = diagonal entries.
%   A = [-1 0 0; -1 -2 0; -1 -1 -3]
%   spectrum = {-1, -2, -3} -> sorted ascending {-3, -2, -1}.
A = [-1 0 0; -1 -2 0; -1 -1 -3];
e = eig(A);
fprintf('%.6f\n', e(1));     % -3.000000
fprintf('%.6f\n', e(2));     % -2.000000
fprintf('%.6f\n', e(3));     % -1.000000

% --- 2. Companion of (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24.
%   Standard companion: row 0..n-2 has unit subdiagonal; last column =
%   negated coefficients. Eigenvalues = roots = {1, 2, 3, 4}.
C = [0 0 0 -24; 1 0 0 50; 0 1 0 -35; 0 0 1 10];
ec = eig(C);
fprintf('%.4f\n', ec(1));    % 1.0000
fprintf('%.4f\n', ec(2));    % 2.0000
fprintf('%.4f\n', ec(3));    % 3.0000
fprintf('%.4f\n', ec(4));    % 4.0000

% --- 3. Rotation matrix — pure-imaginary eigenvalues.
%   A = [0 1; -1 0] -> spectrum = {+i, -i}.
%   eig() returns complex.  disp(real(.)), disp(imag(.)) to render the
%   parts; complex-scalar indexing eig_result(k) trips a separate
%   Sema-side gap so we render whole columns instead.
%   (Build with 0 - 1 instead of -1 in the literal to sidestep the
%   matrix-literal unary-minus lowering gap.)
R = [0 1; 0-1 0];
er = eig(R);
disp(real(er));
disp(imag(er));
% Expected (sorted ascending by re=0, then by im):
%   real = [0; 0]
%   imag = [-1; +1]

% --- 4. Damped-oscillator state matrix (a real-control example).
%   A = [0 1; -wn^2  -2*zeta*wn]  with wn = 2, zeta = 0.5.
%   Eigenvalues: s^2 + 2*zeta*wn*s + wn^2 = 0 ->
%     s = -zeta*wn +- j*wn*sqrt(1 - zeta^2)  for zeta < 1.
%   Numbers: -1 +- j*sqrt(3)  (wn*sqrt(0.75) = 1.7320508).
zeta = 0.5;
wn   = 2.0;
A2   = [0 1; 0-wn*wn, 0-2*zeta*wn];
es   = eig(A2);
disp(real(es));
disp(imag(es));

% --- 5. Symmetric matrix path is unchanged (still Jacobi-fast, real return).
S = [4 1; 1 3];
disp(eig(S));
