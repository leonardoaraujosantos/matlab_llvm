% Fixed-Point Designer (`fi`) — applying a constant gain to a signal sample.
%
% Demonstrates Phase-1 fi support:
%   - constructor with explicit (signed, WL, FL) — Q8.8 here
%   - scalar arithmetic (multiply) on fi values
%   - the `lhs(:) = rhs` clamp idiom that holds the destination spec
%   - disp of an fi value renders the real-world double
%
% See docs/emit_fixed_point.md for the lowering rules.

x = fi(0.75, 1, 16, 8);          % stored = 192
gain = fi(1.5, 1, 16, 8);        % stored = 384
y = fi(0, 1, 16, 8);
y(:) = x * gain;                 % real-world 1.125
disp(y);
