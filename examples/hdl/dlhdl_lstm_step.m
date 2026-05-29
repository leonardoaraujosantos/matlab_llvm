% dlhdl_lstm_step.m — Deep Learning HDL Tier-H4 sequel: a single-input
% LSTM step DUT that holds its own hidden + cell state across cycles via
% MATLAB `persistent` variables.  The two registers (h_state, c_state)
% lower through the EmitSV persistent-register pattern (same shape the
% FIR-pipelined fixture uses for its delay line) — an always_comb block
% computes the next-state expressions, and an always_ff samples them on
% the positive clock edge with async-low reset.
%
% At cycle t the caller drives `x_t`; the DUT returns `h_t = LSTM(x_t,
% h_{t-1}, c_{t-1})` combinationally, and the new (h_t, c_t) are
% registered for cycle t+1.  Compared to dlhdl_lstm_cell.m (combinational
% cell where the caller threaded prev-state through ports), this version
% closes the recurrent loop inside the DUT — the user only needs to clock
% the module, hold `reset = 1` for the first cycle, then stream inputs.
%
% Same hardsigmoid/hardtanh PWL approximations as `dlhdl_lstm_cell.m` for
% all 4 gates + the cell-state activation; same Q16.8 baked weights.
% Module-load driver — necessary so Sema-mono (gated by precise_fi)
% specialises the function with concrete arg types.  We immediately
% follow with a `reset = true` call so the cycle-0 cocotb compare
% starts with both SV (via rst_n) and Python reference (via the early-
% return on reset) at the same (h_state, c_state) = (0, 0) state.
T = numerictype(1, 16, 8);
y = dlhdl_lstm_step(fi(0, T), true);
disp(y);

function h_out = dlhdl_lstm_step(x, reset)
    %#codegen
    % hdl: port(x, fi, signed, 16, 8)
    % hdl: port(reset, bool)
    % hdl: precise_fi
    % cocotb: latency(0)
    % cocotb: range(x, -1.0, 1.0)
    % cocotb: stimulus(reset, impulse, 1)

    % Persistent state registers — h_state and c_state hold the
    % previous cycle's hidden and cell values.  `isempty || reset`
    % drives the always_ff reset path.
    persistent h_state
    persistent c_state
    if isempty(h_state) || reset
        h_state = fi(0, 1, 16, 8);
        c_state = fi(0, 1, 16, 8);
    end

    % Q16.8 baked LSTM weights.
    Wxi = fi( 0.500, 1, 16, 8); Whi = fi( 0.250, 1, 16, 8); bi = fi( 0.125, 1, 16, 8);
    Wxf = fi( 0.375, 1, 16, 8); Whf = fi( 0.125, 1, 16, 8); bf = fi( 0.250, 1, 16, 8);
    Wxg = fi( 0.625, 1, 16, 8); Whg = fi( 0.250, 1, 16, 8); bg = fi( 0.000, 1, 16, 8);
    Wxo = fi( 0.500, 1, 16, 8); Who = fi( 0.375, 1, 16, 8); bo = fi( 0.125, 1, 16, 8);

    one  = fi( 1.000, 1, 16, 8);
    neg1 = fi(-1.000, 1, 16, 8);
    zero = fi( 0.000, 1, 16, 8);
    half = fi( 0.500, 1, 16, 8);

    % i_gate = hardsigmoid(zi) = clamp(0.5*zi + 0.5, 0, 1)
    zi = Wxi * x + Whi * h_state + bi;
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
    zf = Wxf * x + Whf * h_state + bf;
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

    % g_gate = hardtanh(zg)
    zg = Wxg * x + Whg * h_state + bg;
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
    zo = Wxo * x + Who * h_state + bo;
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

    % Cell + hidden update.  c_new uses the previous c_state; h_new
    % feeds through hardtanh on c_new times o_gate.
    c_new = f_gate * c_state + i_gate * g_gate;
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

    % Persistent-state writeback — these become the always_ff `<=`
    % assignments to h_state / c_state on the next positive clk edge.
    h_state = fi(h_new, 1, 16, 8);
    c_state = fi(c_new, 1, 16, 8);

    % Combinational output for the current cycle.
    h_out = fi(h_new, 1, 16, 8);
end
