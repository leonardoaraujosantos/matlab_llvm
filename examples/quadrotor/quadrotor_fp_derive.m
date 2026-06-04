% examples/quadrotor/quadrotor_fp_derive.m
% ---------------------------------------------------------------------------
% First-principles symbolic derivation of the quadrotor 6-DOF equations of
% motion with the Symbolic Math Toolbox (SymPP backend), worked all the way
% down to the decoupled linear models used by quadrotor_fp_flight.m.
%
% This is the "show your work" companion to the flight demo.  It derives, in
% order:
%   1. the body->world rotation matrix R = Rz(psi) Ry(theta) Rx(phi);
%   2. the nonlinear translational dynamics  m*a = R*[0;0;u1] - [0;0;m g];
%   3. the nonlinear rotational dynamics     I*wdot = tau - w x (I w);
%   4. the rotor -> (thrust, torque) mixing matrix;
%   5. the hover linearisations that give each controller its plant:
%        OUTER MPC (x,y):  x_ddot =  g*theta,   y_ddot = -g*phi
%        INNER MPC (z):    z_ddot =  du1/m
%        INNER PID (att):  phi_ddot = u2/Ix, theta_ddot = u3/Iy, psi_ddot = u4/Iz
%
% Run (requires a SymPP-enabled build):
%   cmake -B build -DMATLAB_LLVM_WITH_SYM=ON
%   cat examples/quadrotor/quadrotor_fp_derive.m | build/matlabc -repl /dev/stdin
%
% State (12): [x y z  xd yd zd  phi theta psi  p q r]
% Inputs (4): u1 collective thrust (body +z), u2/u3/u4 body torques (x/y/z).

disp('===============================================================');
disp(' Quadrotor 6-DOF -- first-principles symbolic EOM');
disp('===============================================================');

syms phi theta psi
syms p q r
syms u1 u2 u3 u4
syms m g Ix Iy Iz
syms l kt kd

% --- 1. Body -> world rotation matrix (ZYX Euler) --------------------------
% R = Rz(psi) * Ry(theta) * Rx(phi).  Thrust acts along the body +z axis, so
% the third column of R is what rotates it into the world frame.
disp(' ');
disp('1) Rotation matrix R = Rz(psi)*Ry(theta)*Rx(phi), column by column.');
disp('   Body x-axis in world (column 1):');
disp('     R11 = cos(psi)cos(theta)');
disp(simplify(cos(psi)*cos(theta)));
disp('     R21 = sin(psi)cos(theta)');
disp(simplify(sin(psi)*cos(theta)));
disp('     R31 = -sin(theta)');
disp(simplify(-sin(theta)));
disp('   Body z-axis in world (column 3, the thrust direction):');
disp('     R13 = cos(psi)sin(theta)cos(phi) + sin(psi)sin(phi)');
disp(simplify(cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)));
disp('     R23 = sin(psi)sin(theta)cos(phi) - cos(psi)sin(phi)');
disp(simplify(sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi)));
disp('     R33 = cos(phi)cos(theta)');
disp(simplify(cos(phi)*cos(theta)));

% --- 2. Translational dynamics: m*a = R*[0;0;u1] - [0;0;m g] ----------------
disp(' ');
disp('2) Translational accelerations (full nonlinear):');
disp('   m*x_ddot = u1 * R13 =');
disp(simplify(u1 * (cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))));
disp('   m*y_ddot = u1 * R23 =');
disp(simplify(u1 * (sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))));
disp('   m*z_ddot = u1 * R33 - m*g =');
disp(simplify(u1 * (cos(phi)*cos(theta))));
disp('   ( ... minus m*g )');

% --- 3. Rotational dynamics: I*wdot = tau - w x (I w) ----------------------
% With diagonal inertia I = diag(Ix,Iy,Iz) the Euler equations are
%   Ix*p_dot = u2 + (Iy - Iz)*q*r
%   Iy*q_dot = u3 + (Iz - Ix)*r*p
%   Iz*r_dot = u4 + (Ix - Iy)*p*q
disp(' ');
disp('3) Rotational accelerations (Euler, diagonal inertia):');
disp('   Ix*p_dot - u2 = (Iy - Iz)*q*r =');
disp(simplify((Iy - Iz)*q*r));
disp('   Iy*q_dot - u3 = (Iz - Ix)*r*p =');
disp(simplify((Iz - Ix)*r*p));
disp('   Iz*r_dot - u4 = (Ix - Iy)*p*q =');
disp(simplify((Ix - Iy)*p*q));
disp('   For small body rates the gyroscopic cross terms vanish to first');
disp('   order, leaving  Ix*phi_ddot=u2, Iy*theta_ddot=u3, Iz*psi_ddot=u4.');

% --- 4. Rotor -> (thrust, torque) mixing -----------------------------------
% Four rotors at arm length l in a + configuration produce thrusts f1..f4
% (kt * omega^2) and reaction torques (kd * omega^2).  The control wrench is
%   u1 = f1 + f2 + f3 + f4
%   u2 = l*(f2 - f4)            (roll  from the y-axis pair)
%   u3 = l*(f3 - f1)            (pitch from the x-axis pair)
%   u4 = kd/kt * (f1 - f2 + f3 - f4)   (yaw from rotor drag)
disp(' ');
disp('4) Rotor mixing (+ configuration), symbolic wrench rows:');
syms f1 f2 f3 f4
disp('   u1 (collective) = f1 + f2 + f3 + f4 =');
disp(simplify(f1 + f2 + f3 + f4));
disp('   u2 (roll)  = l*(f2 - f4) =');
disp(simplify(l*(f2 - f4)));
disp('   u3 (pitch) = l*(f3 - f1) =');
disp(simplify(l*(f3 - f1)));
disp('   u4 (yaw)   = (kd/kt)*(f1 - f2 + f3 - f4) =');
disp(simplify((kd/kt)*(f1 - f2 + f3 - f4)));

% --- 5. Hover linearisation -> the controllers' plants ---------------------
% Trim: phi = theta = psi = 0, u1 = m*g.  Linearising the translational
% accelerations about that point decouples the channels.
disp(' ');
disp('5) Hover linearisation (phi=theta=psi=0, u1=m*g):');
disp('   d(x_ddot)/d(theta) at hover  ->  x_ddot = g*theta :');
disp(simplify(diff(g*sin(theta), theta)));
disp('   d(y_ddot)/d(phi)   at hover  ->  y_ddot = -g*phi :');
disp(simplify(diff(-g*sin(phi), phi)));
disp('   altitude with du1 = u1 - m*g  ->  z_ddot = du1/m');
disp('   attitude (small rates)        ->  phi_ddot = u2/Ix, etc.');

disp(' ');
disp('---------------------------------------------------------------');
disp('Cascade used by quadrotor_fp_flight.m:');
disp('  OUTER MPC  : [x;xd;y;yd], in [theta_cmd;phi_cmd], from x_ddot=g*theta');
disp('               and y_ddot=-g*phi.');
disp('  INNER MPC  : [z;zd], in du1, from z_ddot=du1/m (altitude tracker).');
disp('  INNER PID  : roll/pitch/yaw, from phi_ddot=u2/Ix, theta_ddot=u3/Iy,');
disp('               psi_ddot=u4/Iz.');
disp(' ');
disp('Numeric parameters (quadrotor_fp_flight.m):');
disp('  m=1.0 kg, g=9.81 m/s^2, Ix=Iy=0.01, Iz=0.02 kg.m^2, Ts=0.05 s.');
disp('Done.  Run quadrotor_fp_flight.m for the closed-loop flight + animation.');
