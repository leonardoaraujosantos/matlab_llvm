% multilevel decomposition + perfect reconstruction across families
x = cos(2*pi*(0:255)/16) + sin(2*pi*(0:255)/64);
[C,L] = wavedec(x, 4, 'db4');
xr = waverec(C, L, 'db4');
fprintf('db4 PR ok: %.0f\n', round(max(abs(x - xr)) * 1e6));
[C,L] = wavedec(x, 4, 'sym8');
fprintf('sym8 PR ok: %.0f\n', round(max(abs(x - waverec(C,L,'sym8'))) * 1e6));
[C,L] = wavedec(x, 4, 'coif3');
fprintf('coif3 PR ok: %.0f\n', round(max(abs(x - waverec(C,L,'coif3'))) * 1e6));
fprintf('C len: %.0f\n', length(C));
fprintf('num levels: %.0f\n', length(L) - 2);
