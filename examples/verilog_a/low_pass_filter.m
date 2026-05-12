% Verilog-A low-pass filter example — 3rd-order Butterworth at
% 100 kHz, designed via the SPT analog prototype.
%
% Transfer function: H(s) = 1 / (s^3/w0^3 + 2 s^2/w0^2 + 2 s/w0 + 1)
% with w0 = 2*pi*100e3.  MATLAB tf-convention coefficients
% (descending power of s).

w0 = 2.0 * 3.141592653589793 * 100.0e3;
w0_sq  = w0 * w0;
w0_cub = w0 * w0 * w0;
num = [1.0];
den = [1.0/w0_cub; 2.0/w0_sq; 2.0/w0; 1.0];

ok = writeVerilogATF(num, den, "low_pass_filter.va");
disp(ok);
% Drop low_pass_filter.va into Spectre / ngspice / Xyce — the
% module instantiates a `laplace_nd(V(in), {1}, {a0, a1, a2, a3})`
% contribution with the right Butterworth coefficients.
