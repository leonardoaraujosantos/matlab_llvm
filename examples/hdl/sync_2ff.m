function y = sync_2ff(async_in, reset)
    %#codegen
    % hdl: port(async_in, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % Classic 2-flop synchronizer. Async input gets registered
    % twice before the rest of the design sees it, suppressing
    % metastability. The pattern is "stage1 = input; stage2 =
    % stage1; output = stage2" — chained register reads. The
    % Python ref's pre-edge snapshot semantics align with the
    % SV non-blocking model so no explicit latency offset is
    % needed: ref(k) and DUT post-sample at k both produce
    % x_{k-1}.

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
