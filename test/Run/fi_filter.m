% Phase-3 fi gating example: scalar FIR filter (script form).
%
% This is the gating shape from docs/emit_fixed_point.md §7.3 — exercises:
%   - fi(zeros(1,4),...)         array zero-init
%   - fi array indexing           h(i), delay_line(i)
%   - vector concat               [x, delay_line(1:end-1)] (Phase 3.5)
%   - persistent runtime          via the script-scope state binding
%   - scalar fi MAC               acc + (delay_line(i) * h(i))
%   - (:) clamp                   acc(:) = ...
%   - reductions                  none here (sum is exercised in fi_sum tests)
%   - narrowing cast              y = fi(acc, 1, 16, 14)
%
% Coefficients are normalised so the response to a unit step settles at
% sum(h) = 1.0 — easy to eyeball in the .stdout golden.
h = [fi(0.25, 1, 16, 14), fi(0.25, 1, 16, 14), ...
     fi(0.25, 1, 16, 14), fi(0.25, 1, 16, 14)];

delay = fi(zeros(1, 4), 1, 16, 14);

% Push three samples in, observe the accumulator.
for k = 1:3
    x = fi(0.5, 1, 16, 14);
    delay = [x, delay(1:3)];
    acc = fi(0, 1, 16, 14);
    for i = 1:4
        acc(:) = acc + delay(i) * h(i);
    end
    disp(acc);
end
