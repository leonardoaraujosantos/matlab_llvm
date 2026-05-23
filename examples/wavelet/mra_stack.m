% mra_stack.m — Wavelet Toolbox Tier-1.
% ----------------------------------------------------------------------
% Multiresolution analysis stack: decompose a two-tone signal to 5 levels
% with db4, pull out the approximation and per-level details, and prove the
% reconstruction is exact (perfect reconstruction).
x = sin(2*pi*(0:1023)/64) + 0.5*cos(2*pi*(0:1023)/16);

[C, L] = wavedec(x, 5, 'db4');
a5 = appcoef(C, L, 'db4', 5);
d1 = detcoef(C, L, 1);
d3 = detcoef(C, L, 3);
fprintf('approx(5) length = %.0f\n', length(a5));
fprintf('detail(1) length = %.0f\n', length(d1));
fprintf('detail(3) length = %.0f\n', length(d3));

% per-level energy distribution
e = wenergy(C, L);
fprintf('approx energy %%   = %.1f\n', e(1));

xr = waverec(C, L, 'db4');
fprintf('perfect reconstruction error = %.2e\n', max(abs(x - xr)));
