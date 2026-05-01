% Phase 1.1.G — int32 pixel math: a 2x4 buffer flowing through the
% saturating arithmetic path with mixed double scalar operands. Verifies
% MATLAB's "intN beats double" rule (the 1.5 / 2.5 / 3.5 scalars are
% rounded half-away-from-zero, then saturated to int32) and the typed
% disp output.

A = int32([100 200 300 400; -100 -200 -300 -400]);
disp(A);

scaled = A .* 2;
disp(scaled);

shifted = A + 1.5;        % 1.5 -> 2 ; A + 2
disp(shifted);

shifted2 = -2.5 + A;      % -2.5 -> -3 ; -3 + A
disp(shifted2);

quotient = A ./ 7;        % round half-away-from-zero division
disp(quotient);

ovf = int32([2147483640, -2147483640]);
disp(ovf + 100);          % saturates at INT32_MAX / -INT32_MIN
disp(ovf - 100);

% Comparison threading: column-wise count of positive cells.
pos_mask = A > 0;
disp(pos_mask);
disp(sum(pos_mask));      % column sums of the 2x4 mask
