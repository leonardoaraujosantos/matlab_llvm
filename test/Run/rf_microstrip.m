% Microstrip line geometry — Hammerstad-Jensen closed-form.
%
% w=0.95mm, h=0.5mm, εr=4.4 (FR-4 board geometry that targets a
% 50 Ω trace).  Verify the closed-form impedance and effective
% permittivity match textbook values.

freqs = [1.0e9; 2.0e9];
s = rfckt_microstrip(0.95e-3, 0.5e-3, 4.4, 0.05, freqs, 50.0);
disp(s.Z0_line);    % ~50.42 Ω
disp(s.Eeff);       % ~3.33 for FR-4 microstrip

% S11 magnitude is small (line is well-matched to the 50 Ω reference).
S11 = tsS11(s);
disp(S11);

% Coaxial cable (RG-58: a=0.45mm, b=1.475mm, εr=2.25):
%   Z0 = (60/sqrt(2.25)) · ln(1.475/0.45) ≈ 47.5 Ω
sc = rfckt_coaxial(0.45e-3, 1.475e-3, 2.25, 0.1, freqs, 50.0);
disp(sc.Z0_line);
