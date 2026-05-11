% cdma_walsh_demo.m — toy CDMA round-trip with Walsh-Hadamard codes.
%
% Two users share a wireless link via orthogonal spreading codes from
% the 8-chip Walsh-Hadamard matrix.  Each user's binary symbol is
% multiplied by their chip-rate spreading code; the chips of both
% users sum on the air; the receiver despreads by correlating with
% each user's code.  Walsh-code orthogonality means the two users do
% not interfere with each other (when synchronised, no fading).

rng(2034);

% --- Spreading codes ---
N_chip = 8;
code_A = walshCode(N_chip, 2);     % user A: Walsh row 2
code_B = walshCode(N_chip, 6);     % user B: Walsh row 6 (orthogonal to row 2)

% Quick check: correlation should be 0 (orthogonal).
fprintf('Walsh codes used (length %.0f):\n', N_chip);
fprintf('  user A: '); disp(code_A');
fprintf('  user B: '); disp(code_B');
% Inner product via norm/dot - use trace(diag(.)) of code_A' * code_B - but
% simpler: norm-difference identity.  Since each code is +/-1, norm^2 = 8.
% Inner product = (||A+B||^2 - ||A-B||^2) / 4 — pull norms (scalars).
n1 = norm(code_A + code_B);
n2 = norm(code_A - code_B);
fprintf('||code_A + code_B||^2 - ||code_A - code_B||^2 = %.0f (orthogonal -> 0)\n', ...
        n1 * n1 - n2 * n2);

% --- Generate 8 BPSK symbols per user ---
N = 8;
data_A = 2 * (randi(2, N, 1) - 1) - 1;   % {-1, +1}
data_B = 2 * (randi(2, N, 1) - 1) - 1;
fprintf('user A bits: '); disp(data_A');
fprintf('user B bits: '); disp(data_B');

% --- Spread each symbol over the code ---
tx_A = zeros(N * N_chip, 1);
tx_B = zeros(N * N_chip, 1);
for k = 1:N
    tx_A((k-1) * N_chip + 1 : k * N_chip) = data_A(k) * code_A;
    tx_B((k-1) * N_chip + 1 : k * N_chip) = data_B(k) * code_B;
end

% --- Air interface: both users sum, then AWGN ---
air = tx_A + tx_B;
rx = awgn(air, 15);                       % 15 dB SNR

% --- Despread with each user's code ---
hat_A = zeros(N, 1);
hat_B = zeros(N, 1);
for k = 1:N
    chunk = rx((k-1) * N_chip + 1 : k * N_chip);
    % Correlation with the code; divide by N_chip to get back to symbol-
    % amplitude scale (each chip carries ±1; the sum is N_chip · symbol).
    corrA = 0;
    corrB = 0;
    for j = 1:N_chip
        corrA = corrA + chunk(j) * code_A(j);
        corrB = corrB + chunk(j) * code_B(j);
    end
    hat_A(k) = corrA / N_chip;
    hat_B(k) = corrB / N_chip;
end
sliced_A = 2 * (hat_A >= 0) - 1;
sliced_B = 2 * (hat_B >= 0) - 1;
fprintf('decoded A  : '); disp(sliced_A');
fprintf('decoded B  : '); disp(sliced_B');

err_A = symerrCount(data_A, sliced_A);
err_B = symerrCount(data_B, sliced_B);
fprintf('user A symbol errors: %.0f / %.0f\n', err_A, N);
fprintf('user B symbol errors: %.0f / %.0f\n', err_B, N);
