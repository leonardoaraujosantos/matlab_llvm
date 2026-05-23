% FFT with non-power-of-2 length takes the Bluestein path. Check the
% real parts match MATLAB's fft([1 2 3 4 5]) = [15 -2.5 -2.5 -2.5 -2.5].
% NB: do NOT round() — the real parts sit exactly on -2.5, so the
% Bluestein path's last-bit rounding (which differs across libm
% implementations) flips round(-2.5 ± eps) between -2 and -3, making the
% golden platform-dependent.  Printing real(y) shows -2.5 everywhere and
% the tolerance-aware comparison absorbs the eps.
x = [1, 2, 3, 4, 5];
y = fft(x);
disp(real(y));
% ifft(fft(x)) recovers x up to rounding.
z = ifft(y);
disp(real(z));
