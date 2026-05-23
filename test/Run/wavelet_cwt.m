% CWT scalogram of a 50 Hz tone — ridge at the right frequency
fs = 1000;
t = (0:1023)/fs;
x = sin(2*pi*50*t);
[wt, f] = cwt(x, fs);
mag = abs(wt);
e = sum(mag, 2);
[~, idx] = max(e);
fprintf('ridge freq: %.0f\n', round(f(idx)));
fprintf('num scales: %.0f\n', length(f));
a = [1 2 4 8];
fq = scal2frq(a, 'morl', 1/fs);
fprintf('scal2frq f1: %.0f\n', round(fq(1)));
fprintf('scal2frq f4: %.0f\n', round(fq(4)));
