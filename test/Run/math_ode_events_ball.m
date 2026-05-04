% ode_events: ball-drop simulation with terminal event.
% Model: dy/dt = -10 (constant downward velocity), y(0) = 100.
% Closed form: y(t) = 100 - 10*t. The event y == 0 fires at t = 10.
%
% Calling form is non-MATLAB (function-handle-in-struct ABI is still
% TBD): ode_events(@f, tspan, y0, @evt) where evt returns the 3x1
% column [value; isterminal; direction]. With isterminal = 1 the
% integrator halts at the event.

f   = @(t,y) -10;
y0  = 100;
evt = @(t,y) [y; 1; -1];

[t, y, te, ye, ie] = ode_events(f, [0 20], y0, evt);

% Exactly one event captured.
disp(numel(te));               % 1

% Event time at t = 10 (within FP rounding).
if abs(te(1) - 10) < 1e-9; disp(1); else; disp(0); end

% Event value ~ 0.
if abs(ye(1)) < 1e-9; disp(1); else; disp(0); end

% Event index is 1 (only one event component).
disp(ie(1));                    % 1

% Integration halted at the event.
if abs(t(end) - 10) < 1e-9; disp(1); else; disp(0); end
if abs(y(end)) < 1e-9; disp(1); else; disp(0); end
