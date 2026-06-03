% regress_dotop_after_digit.m — regression test for a digit immediately
% followed by a dotted operator (#173). `2.^x` was mis-lexed as the float
% `2.` plus matrix-power `^` (→ unconverted matlab.matpow for a vector RHS),
% instead of `2` element-wise-power `.^` `x`. The lexer now stops the number
% before a `.` that begins a dotted operator, and scalar `.^` (epow of two
% scalars) routes to libm pow. Element-wise scalar disps for order-sensitive,
% backend-independent output.

% --- scalar .^ after a digit ---------------------------------------
disp(3.^2);        % 9
disp(10.^2);       % 100
disp(2.5.^2);      % 6.25
disp(2.^3);        % 8

% --- scalar .^ vector (the headline bug) ---------------------------
w = 2.^[1 2 3];    % [2 4 8]
disp(w(1));        % 2
disp(w(2));        % 4
disp(w(3));        % 8

% --- 2.* / 2./ after a digit still work ----------------------------
u = 2.*[1 2 3];    % [2 4 6]
disp(u(3));        % 6
v = 2./[2 4];      % [1 0.5]
disp(v(1));        % 1

% --- ordinary floats must still lex --------------------------------
disp(2.5);         % 2.5
disp(1.5 + 2.5);   % 4
disp(3.);          % 3

% --- vec .^ scalar / vec .^ vec unaffected -------------------------
a = [1 2 3].^2;    % [1 4 9]
disp(a(3));        % 9
b = [1 2 3].^[0 3 4];  % [1 8 81]
disp(b(3));        % 81
