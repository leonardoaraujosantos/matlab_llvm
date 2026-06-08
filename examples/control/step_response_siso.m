% step(sys) — unit-step response of a SISO LTI system.
%
% Tier 2.3 (control_toolbox_roadmap.md §3.3) — NOT YET SHIPPED.
% Depends on Tier-1.3 (expm) for the closed-form continuous-time path
% and Tier-2.1 (tf / ss model objects) for the input shape.
%
% step internally uses x(t+dt) = expm(A*dt)*x(t) + (closed-form Bd*u)
% so the response is exact at the sample grid (no ode-integrator
% truncation error). Discrete plants use the direct recurrence
% x_{k+1} = A*x_k + B*u_k.

% --- 1. First-order lowpass.  G(s) = 1 / (tau*s + 1), tau = 0.5.
%   Step response:  y(t) = 1 - exp(-t / tau).
tau = 0.5;
G  = tf(1, [tau 1]);
t  = 0 : 0.01 : 3;
[y, tout] = step(G, t);
% At t = tau, y should be (1 - 1/e) ≈ 0.6321.
idx = round(tau / 0.01) + 1;
disp('y(t=tau):');
disp(y(idx));         % ≈ 0.6321

% --- 2. Second-order underdamped.
%   G(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2),  wn = 1, zeta = 0.5.
%   Peak overshoot = exp(-pi*zeta / sqrt(1 - zeta^2)) ≈ 0.1630, so the
%   peak value is ≈ 1.1630.
wn   = 1.0;
zeta = 0.5;
G2   = tf(wn^2, [1, 2*zeta*wn, wn^2]);
t2   = 0 : 0.05 : 15;
y2   = step(G2, t2);
disp('peak of underdamped step:');
disp(max(y2));        % ≈ 1.163
disp('final value of underdamped step:');
disp(y2(end));        % converges to 1

% --- 3. Step-response info.
S = stepinfo(y2, t2);
disp('rise time:');
disp(S.RiseTime);
disp('settling time (2%):');
disp(S.SettlingTime);
disp('overshoot %:');
disp(S.Overshoot);

% --- 4. State-space plant — the same response, different
% representation. Double integrator:  xdot = [0 1; 0 0] x + [0; 1] u,
% y = [1 0] x.  G(s) = 1/s^2.
A = [0 1; 0 0];
B = [0; 1];
C = [1 0];
D = 0;
sys = ss(A, B, C, D);
t3  = 0 : 0.1 : 5;
y3  = step(sys, t3);
% y(t) = t^2 / 2  for the double integrator.
disp('double-integrator y(t=2):');
disp(y3(21));         % 2^2 / 2 = 2.0

% ----- plot the underdamped second-order step response ---------------
figure; plot(t2, y2, 'b-'); grid on;
xlabel('t (s)'); ylabel('y'); title('underdamped step (\zeta=0.5)');
saveas(gcf, '/tmp/ctrl_step.png');
