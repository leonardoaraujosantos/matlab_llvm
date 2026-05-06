% Synthesis-typed call site for examples/hdl/mealy_fsm.m. Run with:
%
%   just emit-sv-multi examples/hdl/mealy_fsm_synth.m \
%                      examples/hdl/mealy_fsm.m
y = mealy_fsm(true, false);
disp(y);
