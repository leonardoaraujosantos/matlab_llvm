% entropy + per-level energy distribution
v = [3 0 0 4];
fprintf('shannon: %.4f\n', wentropy(v, 'shannon'));
fprintf('norm: %.4f\n', wentropy(v, 'norm'));
x = cos(2*pi*(0:255)/16);
[C,L] = wavedec(x, 3, 'db2');
e = wenergy(C, L);
fprintf('energy entries: %.0f\n', length(e));
fprintf('energy sums to 100: %.0f\n', round(sum(e)));
