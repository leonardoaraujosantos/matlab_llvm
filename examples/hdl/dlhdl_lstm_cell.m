% dlhdl_lstm_cell.m — Deep Learning HDL Tier-H4: a single-neuron LSTM cell
% forward pass compiled to fi-typed SystemVerilog via the precise_fi
% opt-in.  All gate non-linearities are HW-friendly piecewise-linear
% surrogates -- `hardsigmoid` (input/forget/output gates) and `hardtanh`
% (cell-candidate gate + final hidden activation).  These are the
% standard substitutes used in quantized LSTM deployments (PyTorch's
% `quantization.lstm`, TensorFlow Lite Micro, etc.) because the exact
% sigmoid/tanh need expensive transcendental approximations on the
% datapath.
%
% Math, per timestep:
%   i = hardsigmoid(Wxi*x + Whi*h_prev + bi)
%   f = hardsigmoid(Wxf*x + Whf*h_prev + bf)
%   g = hardtanh   (Wxg*x + Whg*h_prev + bg)
%   o = hardsigmoid(Wxo*x + Who*h_prev + bo)
%   c_new = f*c_prev + i*g
%   h_new = o * hardtanh(c_new)
%
% The DUT is combinational; the caller threads h_prev/c_prev across
% timesteps externally.  Multi-timestep recurrence with persistent
% registers is a follow-on slice.
T = numerictype(1, 16, 8);
y = dlhdl_lstm_cell(fi(0.5, T), fi(0.25, T), fi(0.125, T));
disp(y);

function h_new = dlhdl_lstm_cell(x, h_prev, c_prev)
    %#codegen
    % hdl: port(x, fi, signed, 16, 8)
    % hdl: port(h_prev, fi, signed, 16, 8)
    % hdl: port(c_prev, fi, signed, 16, 8)
    % hdl: precise_fi
    % cocotb: latency(0)
    % cocotb: range(x, -1.0, 1.0)
    % cocotb: range(h_prev, -1.0, 1.0)
    % cocotb: range(c_prev, -1.0, 1.0)

    % Q16.8 baked weights (in a trained dlnet these come from
    % `dlquantize` applied to the LSTM gate matrices).
    Wxi = fi( 0.500, 1, 16, 8); Whi = fi( 0.250, 1, 16, 8); bi = fi( 0.125, 1, 16, 8);
    Wxf = fi( 0.375, 1, 16, 8); Whf = fi( 0.125, 1, 16, 8); bf = fi( 0.250, 1, 16, 8);
    Wxg = fi( 0.625, 1, 16, 8); Whg = fi( 0.250, 1, 16, 8); bg = fi( 0.000, 1, 16, 8);
    Wxo = fi( 0.500, 1, 16, 8); Who = fi( 0.375, 1, 16, 8); bo = fi( 0.125, 1, 16, 8);

    one  = fi( 1.000, 1, 16, 8);
    neg1 = fi(-1.000, 1, 16, 8);
    zero = fi( 0.000, 1, 16, 8);
    half = fi( 0.500, 1, 16, 8);

    % i_gate = hardsigmoid(zi) = clamp(0.5*zi + 0.5, 0, 1)
    zi = Wxi * x + Whi * h_prev + bi;
    ti = half * zi + half;
    if ti > one
        i_gate = fi(1.0, 1, 16, 8);
    else
        if ti < zero
            i_gate = fi(0.0, 1, 16, 8);
        else
            i_gate = fi(ti, 1, 16, 8);
        end
    end

    % f_gate = hardsigmoid(zf)
    zf = Wxf * x + Whf * h_prev + bf;
    tf_v = half * zf + half;
    if tf_v > one
        f_gate = fi(1.0, 1, 16, 8);
    else
        if tf_v < zero
            f_gate = fi(0.0, 1, 16, 8);
        else
            f_gate = fi(tf_v, 1, 16, 8);
        end
    end

    % g_gate = hardtanh(zg) = clamp(zg, -1, 1)
    zg = Wxg * x + Whg * h_prev + bg;
    if zg > one
        g_gate = fi(1.0, 1, 16, 8);
    else
        if zg < neg1
            g_gate = fi(-1.0, 1, 16, 8);
        else
            g_gate = fi(zg, 1, 16, 8);
        end
    end

    % o_gate = hardsigmoid(zo)
    zo = Wxo * x + Who * h_prev + bo;
    to_v = half * zo + half;
    if to_v > one
        o_gate = fi(1.0, 1, 16, 8);
    else
        if to_v < zero
            o_gate = fi(0.0, 1, 16, 8);
        else
            o_gate = fi(to_v, 1, 16, 8);
        end
    end

    % Cell update: c_new = f * c_prev + i * g
    c_new = f_gate * c_prev + i_gate * g_gate;

    % h_new = o_gate * hardtanh(c_new)
    if c_new > one
        ct = fi(1.0, 1, 16, 8);
    else
        if c_new < neg1
            ct = fi(-1.0, 1, 16, 8);
        else
            ct = fi(c_new, 1, 16, 8);
        end
    end

    h_new = o_gate * ct;
end
