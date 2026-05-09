% Tier 3 — `bandwidth_ss(A, B, C, D)` SISO −3 dB bandwidth.
% Returns the lowest frequency w (rad/s) where |H(jw)| < |H(j0)|/√2.
% Sits on dcgain_ss + bode_ss; scans 200 log-spaced points 1e-3 … 1e6.

% --- 1. First-order lowpass H(s) = 1/(s+1). Closed form: BW = 1.
A = [0-1];
B = [1];
C = [1];
D = [0];
fprintf('1st-order BW (closed form 1.0): %.6f\n', bandwidth_ss(A, B, C, D));

% --- 2. Second-order: wn = 10, zeta = 0.7. Closed form:
% wb = wn · √(1 − 2 ζ² + √((1 − 2 ζ²)² + 1))
%    = 10 · √(0.02 + √(0.0004 + 1))
%    ≈ 10 · 1.009 ≈ 10.09.
wn = 10; zeta = 0.7;
A2 = [0, 1; 0-wn*wn, 0-2*zeta*wn];
B2 = [0; wn*wn];
C2 = [1, 0];
D2 = [0];
fprintf('2nd-order BW (closed form ~10.09): %.4f\n', ...
        bandwidth_ss(A2, B2, C2, D2));

% --- 3. Lower-damping plant — bandwidth shifts higher (resonant peak).
zeta3 = 0.1;
A3 = [0, 1; 0-wn*wn, 0-2*zeta3*wn];
fprintf('low-damping BW (zeta=0.1): %.4f\n', ...
        bandwidth_ss(A3, B2, C2, D2));

% --- 4. Integrator → DC gain unbounded → +Inf return.
A4 = [0];
fprintf('integrator BW (+Inf expected): %.4f\n', bandwidth_ss(A4, B, C, D));
