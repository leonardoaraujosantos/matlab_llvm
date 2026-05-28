% Deep Learning HDL Tier-H2 — emit a quantized 2-2-1 MLP forward pass as
% fi-typed SystemVerilog through the existing EmitSV lane.  The weights
% below are the Q16.8 fixed-point form of a small classifier (the values
% would normally come from `dlquantize` applied to a trained dlnet) and
% are baked into the body as fi constants.  The forward is fully unrolled
% (no per-neuron loop) so the SV emitter sees a fixed dataflow graph the
% saturating-add helpers can be generated for.
T = numerictype(1, 16, 8);
y = dlhdl_quant_mlp(fi(0.5, T), fi(0.25, T));
disp(y);

function y = dlhdl_quant_mlp(x0, x1)
    %#codegen
    % hdl: port(x0, fi, signed, 16, 8)
    % hdl: port(x1, fi, signed, 16, 8)
    % cocotb: latency(0)
    % Layer-1 weights (2 -> 2) — quantized to Q16.8.
    w1_00 = fi( 0.500, 1, 16, 8);
    w1_01 = fi(-0.250, 1, 16, 8);
    w1_10 = fi(-0.375, 1, 16, 8);
    w1_11 = fi( 0.625, 1, 16, 8);
    b1_0  = fi( 0.125, 1, 16, 8);
    b1_1  = fi(-0.125, 1, 16, 8);
    % Layer-2 weights (2 -> 1) — Q16.8.
    w2_0  = fi( 0.750, 1, 16, 8);
    w2_1  = fi(-0.500, 1, 16, 8);
    b2    = fi( 0.250, 1, 16, 8);
    zero  = fi(0, 1, 16, 8);

    % Layer-1 forward + ReLU (hand-unrolled).
    z1_0 = w1_00 * x0 + w1_01 * x1 + b1_0;
    z1_1 = w1_10 * x0 + w1_11 * x1 + b1_1;
    if z1_0 < zero
        h0 = zero;
    else
        h0 = z1_0;
    end
    if z1_1 < zero
        h1 = zero;
    else
        h1 = z1_1;
    end

    % Layer-2 forward (linear output).
    y = w2_0 * h0 + w2_1 * h1 + b2;
end
