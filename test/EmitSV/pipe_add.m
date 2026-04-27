% Phase 5.2 SV — port pipelining via `% hdl: input_pipeline(N)`
% and `% hdl: output_pipeline(N)` pragmas.
%
% input_pipeline(N) adds N register stages on every input port;
% the body's references are routed through the last-stage signal
% (`a_d2` for 2-stage). output_pipeline(N) adds N stages between
% the body's combinational output and the actual port; the
% always_comb writes to a `<port>_d0` pre-pipeline signal, the
% always_ff shifts it through `_d1, _d2, ..., _dN`, and an
% `assign port = <port>_dN` drives the port.
%
% Mirrors HDL Coder's "Input/Output Pipelining = N" project
% option. Adaptive distributed pipelining (cell-cost-driven
% register insertion across combinational depth) is a v2
% follow-up.
T = numerictype(1, 16, 0);
y = pipe_add(fi(3, T), fi(4, T));
disp(y);

function y = pipe_add(a, b)
    %#codegen
    % hdl: input_pipeline(2)
    % hdl: output_pipeline(1)
    y = a + b;
end
