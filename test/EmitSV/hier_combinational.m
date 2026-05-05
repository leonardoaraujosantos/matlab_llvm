function y = hier_combinational(a, b, c, d)
    %#codegen
    % hdl: port(a, fi, signed, 16, 0)
    % hdl: port(b, fi, signed, 16, 0)
    % hdl: port(c, fi, signed, 16, 0)
    % hdl: port(d, fi, signed, 16, 0)
    %
    % Hierarchical multi-module: top instantiates two add2 helpers,
    % then sums the results. Demonstrates the basic instantiation
    % path — no clk/rst_n needed since neither the top nor the
    % helper has persistent state.

    s1 = add2(a, b);
    s2 = add2(c, d);
    y = s1 + s2;
end

function s = add2(x, y)
    %#codegen
    s = x + y;
end
