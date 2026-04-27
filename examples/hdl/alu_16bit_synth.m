% Synthesis-typed call site for examples/hdl/alu_16bit.m. The
% SystemVerilog emitter needs concrete port types; this file
% provides them via a single typed `fi(...)` invocation. Run with:
%
%   just emit-sv-multi examples/hdl/alu_16bit_synth.m \
%                      examples/hdl/alu_16bit.m
T = numerictype(1, 16, 0);
S = numerictype(0, 8, 0);
[d, o] = alu_16bit(fi(5, T), fi(3, T), fi(2, S));
disp(d);
