function y = median3(a, b, c)
    %#codegen
    % hdl: port(a, fi, signed, 16, 0)
    % hdl: port(b, fi, signed, 16, 0)
    % hdl: port(c, fi, signed, 16, 0)
    %
    % 3-input median filter — combinational. Computes the middle
    % value of {a, b, c} using compare-and-select; synthesizes to
    % a small comparator + mux network. The standard "max of mins"
    % identity:
    %   median(a, b, c) = max(min(a, b), min(b, c), min(a, c))
    %
    % Decomposed into pairwise selects since MATLAB's `min`/`max`
    % on scalars lower to runtime calls. Tests:
    %   - signed scalar comparison + select chain
    %   - 6 pairwise compares feeding 3 mins, then a 3-way max
    %   - all combinational (no clock)

    % Pairwise mins (a vs b, b vs c, a vs c).
    if a < b
        ab_min = a;
    else
        ab_min = b;
    end
    if b < c
        bc_min = b;
    else
        bc_min = c;
    end
    if a < c
        ac_min = a;
    else
        ac_min = c;
    end

    % 3-way max of the three mins.
    if ab_min > bc_min
        m1 = ab_min;
    else
        m1 = bc_min;
    end
    if m1 > ac_min
        y = m1;
    else
        y = ac_min;
    end
end
