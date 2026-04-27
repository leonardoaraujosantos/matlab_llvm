% Phase 1 SV — single-add scalar combinational module.
% Two int16 inputs, one int16 output, no control flow.
T = numerictype(1, 16, 0);
y = add_scalar(fi(3, T), fi(4, T));
disp(y);

function y = add_scalar(a, b)
    y = a + b;
end
