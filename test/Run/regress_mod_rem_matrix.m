% regress_mod_rem_matrix.m — regression test for element-wise mod/rem on
% vectors/matrices (#171). Before the fix only the two-scalar form was wired;
% a vector/matrix operand failed with "unsupported call shape". Added _mm /
% _ms / _sm variants (each element through the scalar helper, preserving the
% MATLAB sign rules). Reduced to sums / element disps for order-sensitive,
% backend-independent output.

% --- mod(vector, scalar) -------------------------------------------
a = mod([3 4 6 7], 2);     % [1 0 0 1]
disp(a(1)); disp(a(2)); disp(a(4));   % 1 0 1
disp(sum(a));              % 2

% --- mod(scalar, vector) -------------------------------------------
b = mod(7, [2 3 6]);       % [1 1 1]
disp(sum(b));              % 3

% --- mod(vector, vector) -------------------------------------------
c = mod([4 5 8], [2 2 3]); % [0 1 2]
disp(c(2)); disp(c(3));    % 1 2

% --- mod sign follows the divisor ----------------------------------
disp(mod([-1 -4], 3) * [1;1]);   % mod(-1,3)+mod(-4,3) = 2+2 = 4

% --- rem(vector, scalar): sign follows the dividend ----------------
r = rem([-1 -4 5], 3);     % [-1 -1 2]
disp(r(1)); disp(r(3));    % -1 2

% --- scalar mod/rem still work -------------------------------------
disp(mod(7, 3));           % 1
disp(rem(7, 3));           % 1
