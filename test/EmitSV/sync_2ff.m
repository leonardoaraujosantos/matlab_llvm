function y = sync_2ff(async_in, reset)
    %#codegen
    % hdl: port(async_in, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % Classic 2-flop synchronizer. Async input gets registered
    % twice before the rest of the design sees it, suppressing
    % metastability. The pattern is "stage1 = input; stage2 =
    % stage1; output = stage2" — sequential register chain in
    % MATLAB's blocking semantics, parallel FFs in the SV.

    persistent stage1; persistent stage2;
    if isempty(stage1) || reset
        stage1 = false;
    end
    if isempty(stage2) || reset
        stage2 = false;
    end

    stage1 = async_in;
    stage2 = stage1;
    y = stage2;
end
