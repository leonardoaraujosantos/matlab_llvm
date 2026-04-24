% FFT round-trip: y = fft(x) and then ifft(y) should recover x.
% Pure-C Cooley-Tukey runtime — exact for power-of-2 lengths modulo
% FP rounding; a few-× slower Bluestein path handles general N.
% Here N=4 hits radix-2 directly; separate test exercises Bluestein.
x = [1, 2, 3, 4];
y = fft(x);
disp(real(y));
disp(imag(y));
disp(abs(y));
