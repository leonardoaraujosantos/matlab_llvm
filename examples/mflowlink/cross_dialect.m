% §17.5 #6 — cross-dialect composition. A control-flow MATLAB
% script invokes a baked signal-flow simulation through the
% `mflowlink_run` builtin and prints the final logged-signal
% values.
%
% Compile + run:
%   ./build/matlabc -emit-c examples/mflowlink/cross_dialect.m > /tmp/x.c
%   runtime/build_and_run.sh examples/mflowlink/cross_dialect.m /tmp/cd
%   /tmp/cd
%
% The build script detects `mflowlink_run` and pulls in
% `runtime/runtime_mflowlink_call.cpp` + the matlab_llvm Flowchart
% static libs that own the loader / SignalFlowLowering /
% MflowLinkSim. The compiled binary runs the .mflow's simulation
% to completion in-process and hands the final logged-signal row
% back to MATLAB code.

logged = mflowlink_run("examples/mflowlink/lowpass.mflow");

fprintf('lowpass logged-signal final values:\n');
disp(logged);
