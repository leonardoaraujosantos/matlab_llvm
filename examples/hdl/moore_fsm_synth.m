% Synthesis-typed call site for examples/hdl/moore_fsm.m. Run with:
%
%   just emit-sv-multi examples/hdl/moore_fsm_synth.m \
%                      examples/hdl/moore_fsm.m
[y, st] = moore_fsm(true, false);
disp(y);
disp(st);
