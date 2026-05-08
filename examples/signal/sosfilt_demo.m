% High-order IIR via cascade-of-biquads — tf2sos + sosfilt for
% improved numerical conditioning over direct filter() on the
% transfer-function form.
%
% Design a 6th-order Butterworth lowpass at 0.2 normalised, convert
% it to an SOS matrix (3 biquads), then apply with sosfilt. Compare
% the result to direct filter() — both should give nearly identical
% output, but the SOS form is more robust to coefficient quantization.

[b, a] = butter(6, 0.2);

% sos is a 3 × 6 matrix: each row is [b0 b1 b2 a0 a1 a2].
sos = tf2sos(b, a);
fprintf('SOS sections: %g\n', size(sos, 1));
fprintf('SOS columns:  %g\n', size(sos, 2));
% Print the first section's coefficients.
disp('section 1:');
disp(sos(1, :));

% Apply both forms to a chirp.
fs = 1000;
t  = (0:1/fs:1-1/fs);
x  = chirp(t, 0, 1, 200);

y_tf  = filter(b, a, x);
y_sos = sosfilt(sos, x);

% The difference between the two outputs is dominated by the small
% Durand-Kerner roundoff in the SOS factorisation; for typical filter
% orders it's ~ 1e-3 relative.
fprintf('output rms (filter):  %.4f\n', rms(y_tf));
fprintf('output rms (sosfilt): %.4f\n', rms(y_sos));
