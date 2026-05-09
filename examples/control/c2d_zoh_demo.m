% c2d(sys, Ts, method) — discretise a continuous-time plant or controller.
%
% Tier 2.2 (control_toolbox_roadmap.md §3.2) — NOT YET SHIPPED.
% The 'zoh' branch uses the augmented-matrix expm trick (Tier 1.3),
% the 'tustin' branch is a pure bilinear substitution (no expm
% needed). c2d is the connective tissue between continuous-time
% controller design and the discrete sampled-data implementation.

% --- 1. Continuous PI controller.
%   C(s) = Kp + Ki / s = (Kp s + Ki) / s.  Kp = 2, Ki = 5.
Kp = 2;
Ki = 5;
C = tf([Kp Ki], [1 0]);
disp('continuous controller:');
disp(C);

% --- 2. ZOH discretisation at Ts = 0.05 s.
%   Cd(z) = Kp + Ki * Ts * z / (z - 1)   (ideal integrator → ZOH form)
Ts = 0.05;
Cd_zoh = c2d(C, Ts, 'zoh');
disp('ZOH-discretised controller:');
disp(Cd_zoh);

% --- 3. Tustin (bilinear) discretisation.
%   Substitution s = (2/Ts) * (z - 1)/(z + 1) gives a different but
%   equally valid Cd. Tustin preserves stability margins better at
%   higher frequencies than ZOH.
Cd_tustin = c2d(C, Ts, 'tustin');
disp('Tustin-discretised controller:');
disp(Cd_tustin);

% --- 4. Verify — at the sampling instants, the ZOH-discretised
% step response of C(s) should equal the original continuous step
% response sampled at multiples of Ts. Apply a unit step to both,
% sample the continuous response.
t      = 0 : Ts : 2;
y_cont = step(C, t);            % continuous, sampled at Ts grid
y_zoh  = step(Cd_zoh, t);       % already on the discrete grid
disp('y(0.5s) — continuous vs ZOH:');
disp(y_cont(11));                % 11th sample = t=0.5
disp(y_zoh(11));

% --- 5. Reverse direction — d2c.
C_back = d2c(Cd_zoh, 'zoh');
disp('d2c(c2d(C)) — round-trip Kp:');
[num, den] = tfdata(C_back);
disp(num(1));                    % should be ≈ Kp = 2
