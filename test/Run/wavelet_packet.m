% wavelet packet decompose / reconstruct + node energy
x = sin(2*pi*(0:255)/8) + 0.5*sin(2*pi*(0:255)/4);
T = wpdec(x, 3, 'db2');
fprintf('packet nodes: %.0f\n', size(T,1));
xr = wprec(T, 'db2');
fprintf('wprec PR ok: %.0f\n', round(max(abs(x - xr)) * 1e6));
e = wenergy(T);
fprintf('energy sums to 100: %.0f\n', round(sum(e)));
node = wpcoef(T, 0);
fprintf('node 0 len: %.0f\n', length(node));
