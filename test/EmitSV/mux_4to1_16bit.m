% Phase 1+SW SV — 4:1 multiplexer driven from a typed script entry.
% Demonstrates that the SystemVerilog emitter routes multi-output
% types when the script call site provides typed (fi) args.
y = mux_4to1_16bit(fi(10, 1, 16, 0), fi(20, 1, 16, 0), fi(30, 1, 16, 0), fi(40, 1, 16, 0), fi(2, 0, 8, 0));
disp(y);

function y = mux_4to1_16bit(in0, in1, in2, in3, sel)
    %#codegen
    y = fi(0, 1, 16, 0);
    switch sel
        case 0
            y = in0;
        case 1
            y = in1;
        case 2
            y = in2;
        case 3
            y = in3;
        otherwise
            y = in0;
    end
end
