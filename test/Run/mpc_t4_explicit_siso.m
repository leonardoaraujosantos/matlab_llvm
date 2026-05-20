% MPC Tier-4 §5.1/5.2 — explicit MPC for a SISO double-integrator.
% Tier-4 headline.  Build the table offline by solving the QP at
% every grid point; mpcmoveExplicit then performs O(grid_size · nx)
% nearest-neighbor lookup — no QP solver needed at run-time, ideal
% for embedded deployment.
%
% Plant: stable 2-state discrete.
A = [0.8, 0.0; 0.0, 0.7];
B = [1; 0.5];
C = [1, 0];
D = [0];
sys_d = ss(A, B, C, D, 0.1);

mpcobj = mpc(sys_d, 5, 2);
mpcobj.umax = [10];
mpcobj.umin = [-10];

% Generate the explicit lookup table over the state-space cube
% [-1, 1]^2 with 5+1 = 6 points per dimension → 36 grid points.
x_lo = [-1; -1];
x_hi = [1; 1];
n_grid = 5;
r = [0.5];     % bake in setpoint 0.5

eobj = generateExplicitMPC(mpcobj, x_lo, x_hi, n_grid, r);

fprintf('explicit MPC grid: nx=%.0f, nu=%.0f, n_grid=%.0f\n', ...
        eobj.nx, eobj.nu, eobj.n_grid);
fprintf('u_table size: %.0f x %.0f\n', size(eobj.u_table, 1), ...
        size(eobj.u_table, 2));

% Run-time MV at a few representative states.
xc0 = [0; 0];                    % at origin
u0 = mpcmoveExplicit(eobj, xc0);
fprintf('u(xc=[0;0]) = %.4f\n', u0(1, 1));

xc1 = [0.5; 0];                  % halfway
u1 = mpcmoveExplicit(eobj, xc1);
fprintf('u(xc=[0.5;0]) = %.4f\n', u1(1, 1));

xc2 = [1.0; 0];                  % at the boundary
u2 = mpcmoveExplicit(eobj, xc2);
fprintf('u(xc=[1;0]) = %.4f\n', u2(1, 1));
