% regress_degree_trig.m — regression test for the degree-argument
% trigonometric builtins.  Before the fix, sind / cosd / tand / asind
% / acosd / atand / atan2d were not registered builtins at all, so any
% program using them failed to resolve.  They now have scalar and
% matrix runtime entries (matlab_sind_s / matlab_sind_m / ...).

% --- scalar forms ------------------------------------------------
if abs(sind(30) - 0.5) < 1e-12; disp(1); else; disp(0); end
if abs(cosd(60) - 0.5) < 1e-12; disp(1); else; disp(0); end
if abs(tand(45) - 1.0) < 1e-12; disp(1); else; disp(0); end
if abs(sind(90) - 1.0) < 1e-12; disp(1); else; disp(0); end
if abs(cosd(0)  - 1.0) < 1e-12; disp(1); else; disp(0); end

% --- inverse forms (result in degrees) ---------------------------
if abs(asind(0.5) - 30) < 1e-9; disp(1); else; disp(0); end
if abs(acosd(0.5) - 60) < 1e-9; disp(1); else; disp(0); end
if abs(atand(1.0) - 45) < 1e-9; disp(1); else; disp(0); end
if abs(atan2d(1.0, 1.0) - 45) < 1e-9; disp(1); else; disp(0); end

% --- matrix forms (element-wise) ---------------------------------
v = sind([0; 30; 90]);
e = abs(v(1) - 0) + abs(v(2) - 0.5) + abs(v(3) - 1);
if e < 1e-12; disp(1); else; disp(0); end

w = cosd([0, 60, 90]);
e2 = abs(w(1) - 1) + abs(w(2) - 0.5) + abs(w(3) - 0);
if e2 < 1e-12; disp(1); else; disp(0); end
