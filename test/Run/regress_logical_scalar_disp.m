% regress_logical_scalar_disp.m — regression test for displaying a scalar
% logical / comparison result (#152). Before the fix, disp of an i1 `true`
% sign-extended to f64 (-1.0) instead of zero-extending (1.0): the LowerIO
% scalar-disp path used SIToFP for any integer and only switched to UIToFP
% on an explicit `matlab.unsigned` tag, which an i1 logical never carries.
% So `disp(5>0)` / `disp(1|0)` printed -1. An i1 is always a 0/1 logical, so
% it now zero-extends.

% --- logical operators ---------------------------------------------
disp(1 | 0);     % 1
disp(1 & 1);     % 1
disp(0 & 1);     % 0
disp(~0);        % 1
disp(~5);        % 0

% --- comparison operators ------------------------------------------
disp(5 > 0);     % 1
disp(3 < 1);     % 0
disp(2 == 2);    % 1
disp(2 ~= 2);    % 0
disp(4 >= 4);    % 1

% --- plain numeric disp is unaffected ------------------------------
disp(5);         % 5
disp(-3);        % -3
