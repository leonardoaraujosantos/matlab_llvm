% Synthesis-typed call site for examples/hdl/counter_0_to_10.m.
% Run with:
%
%   just emit-sv-multi examples/hdl/counter_0_to_10_synth.m \
%                      examples/hdl/counter_0_to_10.m
reset = false;
c = counter_0_to_10(reset);
disp(c);
