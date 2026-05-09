% Tier 3 (CST roadmap §4) — model characterization triad.
%   isstable(A): 1 if Hurwitz, else 0.
%   damp(A):     n x 2 table [wn, zeta] per pole.
%   hsvd(A,B,C): Hankel singular values sqrt(eig(Wc Wo)) sorted desc.

% --- 1. Hurwitz (stable) plant.
A = [0-1, 0; 0, 0-2];
disp('isstable(diag(-1, -2)):');
disp(isstable(A));     % 1

% --- 2. Unstable plant.
A2 = [1, 0; 0, 0-1];
disp('isstable([1 0; 0 -1]):');
disp(isstable(A2));    % 0

% --- 3. Marginal (eigvals on imaginary axis): rotation matrix.
A3 = [0, 1; 0-1, 0];
disp('isstable([0 1; -1 0]):');
disp(isstable(A3));    % 0 — marginal isn't stable in MATLAB

% --- 4. damp on real pole.
disp('damp([-2]) — wn=2, zeta=1:');
disp(damp([0-2]));

% --- 5. damp on underdamped 2nd order.
% A = [0 1; -wn^2 -2*zeta*wn]; wn = 2, zeta = 0.5.
wn = 2.0; zeta = 0.5;
A4 = [0, 1; 0-wn*wn, 0-2*zeta*wn];
disp('damp(2nd order, wn=2, zeta=0.5) — both rows = [2, 0.5]:');
disp(damp(A4));

% --- 6. hsvd on a 1st-order Hurwitz plant: A=-1, B=1, C=1.
% Wc = Wo = 1/2, Wc*Wo = 1/4, hsv = 1/2.
disp('hsvd(-1, 1, 1) — closed form 1/2:');
disp(hsvd([0-1], [1], [1]));

% --- 7. hsvd on a 2nd-order plant — diagonal structure with two
% distinct decay rates and full coupling.
A5 = [0-1, 0; 0, 0-2];
B5 = [1; 1];
C5 = [1, 1];
disp('hsvd(diag(-1,-2), [1;1], [1 1]):');
disp(hsvd(A5, B5, C5));
