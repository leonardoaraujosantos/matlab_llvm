% dlhdl_rnn_cell.m — Deep Learning HDL Tier-H4 first slice: a simple
% recurrent cell `h_new = hardtanh(Wx*x + Wh*h_prev + b)` compiled to
% fi-typed SystemVerilog via the precise_fi opt-in.  The user threads
% the cell across timesteps (h_prev is an input port; the DUT is
% combinational).  This proves the recurrent kernel can compile to
% bit-accurate fi SV before we add the full LSTM gate machinery.
T = numerictype(1, 16, 8);
y = dlhdl_rnn_cell(fi(0.5, T), fi(0.25, T));
disp(y);

function h_new = dlhdl_rnn_cell(x, h_prev)
    %#codegen
    % hdl: port(x, fi, signed, 16, 8)
    % hdl: port(h_prev, fi, signed, 16, 8)
    % hdl: precise_fi
    % cocotb: latency(0)
    % cocotb: range(x, -1.0, 1.0)
    % cocotb: range(h_prev, -1.0, 1.0)
    % Q16.8 baked weights (in a trained dlnet these would come from
    % `dlquantize` applied to the recurrent kernel).
    Wx   = fi( 0.500, 1, 16, 8);
    Wh   = fi( 0.250, 1, 16, 8);
    b    = fi( 0.125, 1, 16, 8);
    one  = fi( 1.000, 1, 16, 8);
    neg1 = fi(-1.000, 1, 16, 8);

    % Affine pre-activation.
    z = Wx * x + Wh * h_prev + b;

    % `hardtanh` — the standard HW-friendly PWL approximation of
    % the LSTM tanh activation.  Lowers to nested muxes in SV.  Each
    % branch is explicitly fi-typed at the function's output Q-spec
    % (Q16.8) so the inferred output type matches the port pragma.
    h_new = fi(0, 1, 16, 8);
    if z > one
        h_new(:) = one;
    else
        if z < neg1
            h_new(:) = neg1;
        else
            h_new(:) = z;
        end
    end
end
