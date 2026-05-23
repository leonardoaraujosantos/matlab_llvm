% MODWT round-trip + MRA additivity
x = sin(2*pi*(0:511)/16) + cos(2*pi*(0:511)/64);
w = modwt(x, 'sym4', 4);
xr = imodwt(w, 'sym4');
fprintf('modwt rows: %.0f\n', size(w,1));
fprintf('imodwt PR ok: %.0f\n', round(max(abs(x - xr)) * 1e6));
mra = modwtmra(w, 'sym4');
s = sum(mra, 1);
fprintf('mra sums to signal: %.0f\n', round(max(abs(x - s)) * 1e6));
