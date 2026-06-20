% #322 regression: the data-returning multi-output forms of the control
% response builtins (no plot) — the companion to ctrl_step_data.m's
% [y,t]=step(sys). Each previously raised "unsupported call shape".
%
%   [mag,phase,w] = bode(sys)     auto log-frequency grid
%   [re,im,w]     = nyquist(sys)  real/imag of the freq response
%   [y,t]         = impulse(sys)  state-space impulse response
%   [y,t]         = initial(sys,x0) free response from x0
sys = tf([1], [1 1]);            % H(s) = 1/(s+1)

[mag, phase, w] = bode(sys);
fprintf('bode_n = %d\n', numel(mag));
fprintf('bode_nw = %d\n', numel(w));
fprintf('bode_mag1 = %.4f\n', mag(1));   % |H(jw)| ~ 1 at low w

[re, im, wn] = nyquist(sys);
fprintf('nyq_n = %d\n', numel(re));
fprintf('nyq_re1 = %.4f\n', re(1));      % ~ 1 at low w
fprintf('nyq_im1 = %.4f\n', im(1));      % ~ 0- at low w

ssys = ss(-1, 1, 1, 0);          % same plant, state-space
[yi, ti] = impulse(ssys);
fprintf('imp_n = %d\n', numel(yi));
fprintf('imp_y0 = %.4f\n', yi(1));       % impulse response h(0) = 1
fprintf('imp_tend = %.2f\n', ti(end));

[yc, tc] = initial(ssys, 2.0);
fprintf('ini_n = %d\n', numel(yc));
fprintf('ini_y0 = %.4f\n', yc(1));       % y(0) = C*x0 = 2
fprintf('ini_yend = %.4f\n', yc(end));   % decays toward 0

% y = lsim(sys, u, t): forced response to input u over time vector t.
tt = (0:0.01:1)';
uu = ones(numel(tt), 1);          % unit step input
yl = lsim(ssys, uu, tt);
fprintf('lsim_n = %d\n', numel(yl));
fprintf('lsim_yend = %.4f\n', yl(end));   % step resp at t=1: 1-e^-1 = 0.6321
