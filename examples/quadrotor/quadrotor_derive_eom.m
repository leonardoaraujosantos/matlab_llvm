% examples/quadrotor/quadrotor_derive_eom.m
% --------------------------------------------------------------------
% Symbolic derivation of the 6-DOF quadrotor equations of motion using
% the Symbolic Math Toolbox (SymPP backend).
%
% State (12 components):
%   - position:        [x, y, z]   in world frame (z up)
%   - linear velocity: [u, v, w]   in world frame
%   - attitude:        [phi, theta, psi]  (roll, pitch, yaw — ZYX Euler)
%   - body rates:      [p, q, r]
%
% Inputs (4 components):
%   - u1: collective thrust along body +z
%   - u2, u3, u4: torques about body x, y, z
%
% Outputs the linearized cascade that the cascade controller uses:
%   Outer (MPC):  x_ddot =  g * theta,   y_ddot = -g * phi
%   Inner (PIDs): z_ddot = du1/m, phi_ddot = u2/Ix, theta_ddot = u3/Iy
%
% Run with:
%   matlabc examples/quadrotor/quadrotor_derive_eom.m
% Requires matlabc built with -DMATLAB_LLVM_WITH_SYM=ON.

disp('========================================================');
disp(' Quadrotor 6-DOF — symbolic equations of motion');
disp('========================================================');

syms phi theta psi
syms u1 u2 u3 u4
syms m g Ix Iy Iz

% --- Translational dynamics --------------------------------------------
% Body→world rotation (ZYX Euler).  The third column of R rotates the
% body z-axis (where thrust acts) into the world frame, giving:

disp(' ');
disp('Translational accelerations (full nonlinear):');
disp('  m * x_ddot =');
disp(simplify(u1 * (cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))));
disp('  m * y_ddot =');
disp(simplify(u1 * (cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))));
disp('  m * z_ddot = u1*cos(phi)*cos(theta) - m*g');

% --- Rotational dynamics -----------------------------------------------
% I * w_dot = tau - w x (I w). For diagonal I and small body rates the
% cross-coupling vanishes to first order:

disp(' ');
disp('Rotational accelerations (diagonal I, small body rates):');
disp('  Ix * phi_ddot   = u2');
disp('  Iy * theta_ddot = u3');
disp('  Iz * psi_ddot   = u4');

% --- Linearize about hover ---------------------------------------------
% Hover trim: phi = theta = psi = 0, u1 = m*g.  Substitute and simplify
% to get the decoupled linear plant used by the outer MPC.

disp(' ');
disp('Linearized about hover (phi=theta=psi=0, u1 = m*g):');

disp('  d/d(theta) of x_ddot at hover (= g):');
disp(simplify(diff(g * sin(theta), theta)));
disp('  -> x_ddot = g * theta');

disp('  d/d(phi) of y_ddot at hover (= -g):');
disp(simplify(diff(-g * sin(phi), phi)));
disp('  -> y_ddot = -g * phi');

disp('  z_ddot with du1 = u1 - m*g:');
disp(simplify((m*g + sym(0)) * cos(sym(0)) * cos(sym(0)) - m*g));
disp('  -> z_ddot = du1/m');

% --- Cascade structure used by the controller --------------------------

disp(' ');
disp('Cascade structure used by the controller:');
disp('  Outer loop (MPC, slow): ');
disp('     plant 1: x_ddot = g * theta         (pitch -> +x)');
disp('     plant 2: y_ddot = -g * phi          (roll  -> -y)');
disp('     state [x; x_dot; y; y_dot], input [theta_cmd; phi_cmd]');
disp('  Inner loop (PIDs, fast):');
disp('     altitude: z_ddot = du1 / m');
disp('     roll:     phi_ddot   = u2 / Ix');
disp('     pitch:    theta_ddot = u3 / Iy');
disp('     yaw:      psi_ddot   = u4 / Iz');

disp(' ');
disp('Numeric model parameters used by quadrotor_pid_mpc.m:');
disp('   m  = 1.0 kg     (vehicle mass)');
disp('   g  = 9.81 m/s^2 (gravity)');
disp('   Ix = Iy = 0.01 kg.m^2,  Iz = 0.02 kg.m^2');
disp('   Ts = 0.05 s     (outer MPC step; inner sub-steps x10)');
disp(' ');
disp('Done.  Run quadrotor_pid_mpc.m for the full simulation.');
