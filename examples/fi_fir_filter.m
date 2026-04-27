% Fixed-Point Designer (`fi`) — scalar FIR moving-average filter at script scope.
%
% Demonstrates Phase-3 fi support:
%   - fi array zero-init (`fi(zeros(1,N), ...)`)
%   - vector concat `[x, delay(1:end-1)]` for the shift register
%   - scalar fi MAC (multiply-accumulate)
%   - the `(:)` clamp idiom that holds the accumulator's spec
%
% The 4-tap moving average has impulse response 1/4 per tap, so the step
% response settles at 1.0 and the partial sums are 0.25, 0.5, 0.75, 1.0.
%
% See docs/emit_fixed_point.md §7.3 for the full lowering shape.

h = [fi(0.25, 1, 16, 14), fi(0.25, 1, 16, 14), ...
     fi(0.25, 1, 16, 14), fi(0.25, 1, 16, 14)];

delay = fi(zeros(1, 4), 1, 16, 14);

for k = 1:4
    x = fi(1.0, 1, 16, 14);
    delay = [x, delay(1:3)];
    acc = fi(0, 1, 16, 14);
    for i = 1:4
        acc(:) = acc + delay(i) * h(i);
    end
    disp(acc);
end
