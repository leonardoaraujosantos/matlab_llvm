% Synthesis-typed call site for examples/hdl/mealy_fsm.m. Run with:
%
%   just emit-sv-multi examples/hdl/mealy_fsm_synth.m \
%                      examples/hdl/mealy_fsm.m
T = numerictype(0, 8, 0);
y = mealy_fsm(fi(1, T), fi(0, T));
disp(y);
