% Deep Learning HDL Tier-H1 gating test — INT8 weight quantization.
%
% Verifies:
%   (a) dlquantize(W) returns a matrix the same shape as W,
%   (b) every quantized weight is on an integer multiple of dlqscale(W),
%   (c) the quantization error is bounded by (scale / 2) — the int8 LSB.

W = [ 1.2 -0.4  0.05  0.0   0.9;
     -1.5  0.3 -0.02  0.7  -0.8;
      0.1  0.6  0.0  -0.3   0.4 ];

Wq = dlquantize(W);
s  = dlqscale(W);
sv = s(1);

shape_ok = 0;
if size(Wq, 1) == size(W, 1)
    if size(Wq, 2) == size(W, 2)
        shape_ok = 1;
    end
end

% Quantized values should be integer multiples of the scale; error per element
% should not exceed half the scale (round-to-nearest on the int8 grid).
max_err = 0;
max_lattice_err = 0;
for i = 1:size(W, 1)
    for j = 1:size(W, 2)
        e = abs(Wq(i, j) - W(i, j));
        if e > max_err; max_err = e; end
        q_idx = round(Wq(i, j) / sv);
        lattice_err = abs(Wq(i, j) - q_idx * sv);
        if lattice_err > max_lattice_err; max_lattice_err = lattice_err; end
    end
end

bounded = 0;
if max_err <= 0.5 * sv + 1e-12
    bounded = 1;
end
lattice_ok = 0;
if max_lattice_err < 1e-9
    lattice_ok = 1;
end

% Scale should equal max(abs(W)) / 127.  max(abs(W)) here is 1.5.
expected_scale = 1.5 / 127.0;
scale_ok = 0;
if abs(sv - expected_scale) < 1e-12
    scale_ok = 1;
end

fprintf('quantize shape ok = %.0f\n', shape_ok);
fprintf('scale matches max(abs(W))/127 = %.0f\n', scale_ok);
fprintf('every weight on int8 lattice = %.0f\n', lattice_ok);
fprintf('quantization error <= scale/2 = %.0f\n', bounded);
