% DSP System Toolbox Tier-1 — System-Object lifecycle.
%
% Exercises reset (zero the DiscreteState), getDiscreteState (read the
% tapped-delay line), and that a second independent object keeps its own
% state (handle objects are distinct instances).  Lifecycle methods use
% dot-dispatch (obj.method()).  Single-sample frames make the persisted
% state visible between calls.
b = [0.5 0.5];
f = dsp.FIRFilter('Numerator', b);

% Feed an impulse one sample at a time.
y1 = f(1);
fprintf('out1 %.2f\n', y1);

% The delay line now holds the spillover (0.5) for the next call.
s = f.getDiscreteState();
fprintf('state %.2f\n', s(1));

% Next zero input flushes the tail.
y2 = f(0);
fprintf('out2 %.2f\n', y2);

% Reset clears the state.
f.reset();
s2 = f.getDiscreteState();
fprintf('state after reset %.2f\n', s2(1));

% A second object is independent of the first.
g = dsp.FIRFilter('Numerator', b);
y3 = g(2);
fprintf('second obj out %.2f\n', y3);
y4 = f(0);
fprintf('first obj still reset, out %.2f\n', y4);
