% Transfer function model objects — tf(num, den).
%
% Tier 2.1 (control_toolbox_roadmap.md §3.1) — NOT YET SHIPPED.
% This file documents the intended API; it will run once the Tier-2.1
% slice (model object constructors as classdef + operator overloads)
% lands.
%
% MATLAB's tf is a value class; matlab_llvm models it with a classdef
% (handle-shaped) where every operator overload returns a fresh
% instance — value semantics fall out for free since CST functions
% never mutate in place.

% --- 1. Construct a SISO continuous-time transfer function.
%   G(s) = (s + 2) / (s^2 + 3s + 5)
G = tf([1 2], [1 3 5]);
disp(G);
% Expected (canonical s-domain rendering):
%
% G(s) =
%
%        s + 2
%   ----------------
%    s^2 + 3 s + 5
%
% Continuous-time transfer function.

% --- 2. The "tf('s')" variable-builder idiom.
%   Compose transfer functions using polynomial-style algebra in s.
s = tf('s');
G2 = (s + 2) / (s^2 + 3*s + 5);
disp(G2);                % same as G above

% --- 3. Operator overloads — interconnections.
%
%   G + H  → parallel combination (sum)
%   G * H  → series cascade (G feeds into H)
%   G / H  → right-divide (G * inv(H))
%   G'     → Hermitian transpose (MIMO swap)
H = tf(1, [1 1]);        % H(s) = 1/(s + 1)
parallel = G + H;        % G(s) + 1/(s+1)
cascade  = G * H;        % G(s) * 1/(s+1)
disp(parallel);
disp(cascade);

% --- 4. Discrete-time transfer functions.
%   The third positional argument is the sample time Ts.
%   G_d(z) = (z - 0.5) / (z^2 - 0.6 z + 0.1) at Ts = 0.01 s.
Gd = tf([1 -0.5], [1 -0.6 0.1], 0.01);
disp(Gd);
% Expected:
%
% Gd(z) =
%
%        z - 0.5
%   --------------------
%    z^2 - 0.6 z + 0.1
%
% Sample time: 0.01 seconds
% Discrete-time transfer function.

% --- 5. Static info.
[num, den] = tfdata(G);
disp('numerator length:');
disp(length(num));
disp('denominator length:');
disp(length(den));

% ----- plot the step response of G(s) = (s+2)/(s^2+3s+5) -------------
ts = 0:0.05:8;
ys = step(G, ts);
figure; plot(ts, ys, 'b-'); grid on;
xlabel('t (s)'); ylabel('y'); title('step response of G(s)');
saveas(gcf, '/tmp/ctrl_tf_step.png');
