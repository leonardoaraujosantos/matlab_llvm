% Tier 4 — balanced truncation workflow.
% balred_A / balred_B / balred_C return the k-state truncated balanced
% realization. The dropped states correspond to the smallest Hankel
% singular values; the H∞ error bound is 2·sum(HSV[k+1:n]).

% --- 4-state plant: dominant 2-state mass-spring-damper plus two
% near-decoupled fast modes. The fast modes have small B/C entries,
% so they barely participate in the I/O map.
A = [0,    1,     0,     0;
     0-9, 0-0.3,  0,     0;
     0,    0,    0-10,    0;
     0,    0,    0,    0-20];
B = [0; 1; 0.001; 0.001];
C = [1, 0, 0.01, 0.01];

% --- Inspect the Hankel singular values: the last two are ~1e-7,
% several orders of magnitude smaller than the first two — clear
% candidates for truncation.
H = hsvd(A, B, C);
disp('Hankel singular values (descending):');
disp(H);
disp('H∞ error bound 2·sum(HSV[3:4]) for k=2 truncation:');
disp(2 * (H(3,1) + H(4,1)));

% --- Truncate to k = 2.
Ar = balred_A(A, B, C, 2);
Br = balred_B(A, B, C, 2);
Cr = balred_C(A, B, C, 2);

disp('reduced (Ar, Br, Cr):');
disp(Ar);
disp(Br);
disp(Cr);

% --- Properties to check on the reduced model:
%  1) Stability preserved (Hurwitz).
disp('isstable(Ar) (must be 1):');
disp(isstable(Ar));

%  2) Dominant HSVs unchanged — the truncated balanced realization's
%     Hankel SVs equal the top-k HSVs of the original.
disp('hsvd(Ar, Br, Cr) — must match top-2 of original:');
disp(hsvd(Ar, Br, Cr));

%  3) Damping ratios of the dominant mode preserved.
disp('damp(A) (full plant):');
disp(damp(A));
disp('damp(Ar) (reduced plant — dominant mode only):');
disp(damp(Ar));

% ----- plot the Hankel SVs (log scale shows the tail to truncate) ----
figure; bar(log10(H + 1e-12)); grid on;
xlabel('state'); ylabel('log_{10} HSV');
title('Hankel SVs: top-2 dominate (truncate to k=2)');
saveas(gcf, '/tmp/ctrl_balred_hsv.png');
