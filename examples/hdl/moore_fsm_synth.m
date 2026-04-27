% Synthesis-typed call site for examples/hdl/moore_fsm.m. Run with:
%
%   just emit-sv-multi examples/hdl/moore_fsm_synth.m \
%                      examples/hdl/moore_fsm.m
T = numerictype(0, 8, 0);
[y, st] = moore_fsm(fi(1, T), fi(0, T));
disp(y);
disp(st);
