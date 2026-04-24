% FFT with non-power-of-2 length takes the Bluestein path. The
% result has tiny imag parts from rounding; round them and check
% that real parts match MATLAB's fft([1 2 3 4 5]).
x = [1, 2, 3, 4, 5];
y = fft(x);
disp(round(real(y)));
% ifft(fft(x)) recovers x up to rounding.
z = ifft(y);
disp(round(real(z)));
