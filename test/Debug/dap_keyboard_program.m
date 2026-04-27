% Fixture for the `keyboard` builtin. Hitting line 5 should pause
% the worker with stop reason="entry"; once resumed (continue), the
% script runs disp(99) and exits.
x = 41;
keyboard;
disp(x);
