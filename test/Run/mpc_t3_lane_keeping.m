% MPC Tier-3 headline — lane-keeping assist (MPC Toolbox User's
% Guide §12.10), simplified.  2-state lateral-dynamics model:
%   x1 = lateral position (lane error), x2 = lateral velocity
%   u  = lateral acceleration command (proxy for steering effort)
%   y  = lateral position
% The MPC steers the car back to lane center (y_lat → 0 after a
% step from 1 m), with a tight ±2 m/s² acceleration bound and
% disturbance integrator for offset-free tracking under wind/bank
% disturbances.  Demonstrates Tier-1 + Tier-2 features combined
% in the canonical AD use case.

A_d = [1.00, 0.05;
       0.00, 0.90];      % position += vel*Ts; vel decays
B_d = [0.00;
       0.05];            % accel input integrates into vel
C_d = [1, 0];
D_d = [0];
sys_d = ss(A_d, B_d, C_d, D_d, 0.05);

obj = mpc(sys_d, 15, 3);
obj.umax = [2.0];
obj.umin = [-2.0];
obj.outdist = 1;
obj.Wy  = [5.0];                  % heavy weight on lane position
obj.Wdu = [0.2];                  % gentle move suppression

% Start with the car 1 m off-lane; reference y = 0 (lane center).
% Use mpcmove tick-by-tick so we can manually inject the initial
% lateral offset via the mpcstate.
st = mpcstate(2, 1, 1);
st.Plant = [1.0; 0.0];            % offset state

T = 60;
r = [0];                          % return to lane center

% Verify a few key timepoints by stepping a fixed-form loop.
ym = [1.0];                       % first measurement = offset
u  = mpcmove(obj, st, ym, r);
fprintf('tick 1  u = %.4f\n', u(1, 1));
fprintf('tick 1  st.Plant(1) = %.4f (lat. pos. after step)\n', st.Plant(1, 1));

% A few more ticks via sim — note: sim restarts from zero state, so
% it confirms the controller behavior for the canonical step.
sys_pos = ss(A_d, B_d, C_d, D_d, 0.05);
obj2 = mpc(sys_pos, 15, 3);
obj2.umax = [2.0]; obj2.umin = [-2.0];
obj2.outdist = 1; obj2.Wy = [5.0]; obj2.Wdu = [0.2];
T2 = 60; r2 = [1.0];              % step from 0 to 1
y = sim(obj2, T2, r2);
fprintf('  t=0.25s  y_lat = %.4f\n', y(5, 1));
fprintf('  t=0.50s  y_lat = %.4f\n', y(10, 1));
fprintf('  t=1.00s  y_lat = %.4f\n', y(20, 1));
fprintf('  t=3.00s  y_lat = %.4f\n', y(60, 1));
