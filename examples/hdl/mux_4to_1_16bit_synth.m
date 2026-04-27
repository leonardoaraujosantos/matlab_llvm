% Synthesis-typed call site for examples/hdl/mux_4to_1_16bit.m.
% Run with:
%
%   just emit-sv-multi examples/hdl/mux_4to_1_16bit_synth.m \
%                      examples/hdl/mux_4to_1_16bit.m
T = numerictype(1, 16, 0);
S = numerictype(0, 8, 0);
y = mux_4to1_16bit(fi(10, T), fi(20, T), fi(30, T), fi(40, T), fi(2, S));
disp(y);
