% single-level DWT round-trip + family-filter sums
x = sin(2*pi*(0:63)/8) + 0.4*cos(2*pi*(0:63)/4);
[cA, cD] = dwt(x, 'db4');
xr = idwt(cA, cD, 'db4');
fprintf('idwt PR ok: %.0f\n', round(max(abs(x - xr)) * 1e6));
fprintf('cA len: %.0f\n', length(cA));
[lod, hid, lor, hir] = wfilters('db4');
fprintf('db4 sum(Lo_D): %.4f\n', sum(lod));
fprintf('db4 sum(Lo_D^2): %.4f\n', sum(lod.^2));
fprintf('db4 filter len: %.0f\n', length(lod));
