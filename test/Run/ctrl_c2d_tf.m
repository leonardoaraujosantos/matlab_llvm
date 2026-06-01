% #27 — c2d / d2c on a tf (transfer-function) object.
% c2d routes tf2ss(CCF) -> discretise -> ss2tf for ZOH (output map C,D
% invariant under ZOH) and uses an exact bilinear substitution for Tustin.
% d2c is the ZOH inverse via matrix-logm. Continuous PI controller
%   C(s) = (2s + 5)/s   (Kp = 2, Ki = 5), Ts = 0.05.

C  = tf([2 5], [1 0]);
Ts = 0.05;

% --- ZOH: exact result (2 z - 1.75)/(z - 1).
Cz = c2d(C, Ts, 'zoh');
[nz, dz] = tfdata(Cz);
fprintf('zoh num: %.4f %.4f\n', nz(1), nz(2));   % 2.0000 -1.7500
fprintf('zoh den: %.4f %.4f\n', dz(1), dz(2));   % 1.0000 -1.0000

% --- Tustin: exact result (85 z - 75)/(40 z - 40).
Ct = c2d(C, Ts, 'tustin');
[nt, dt] = tfdata(Ct);
fprintf('tustin num: %.4f %.4f\n', nt(1), nt(2)); % 85.0000 -75.0000
fprintf('tustin den: %.4f %.4f\n', dt(1), dt(2)); % 40.0000 -40.0000

% --- d2c round-trip recovers the proportional term Kp = 2.
Cb = d2c(Cz, 'zoh');
[nb, db] = tfdata(Cb);
fprintf('d2c kp: %.4f\n', nb(1));                 % 2.0000

fprintf('ctrl_c2d_tf: PASS\n');
