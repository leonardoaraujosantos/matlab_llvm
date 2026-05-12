% Smoke: rfAnalyze* helpers that synthesize S-parameter structs from
% rfckt block scalar properties.

freqs = [1.0e9; 2.0e9];

% Amplifier with 20 dB gain at z0 = 50 Ω.
s_amp = rfAnalyzeAmplifier(20.0, freqs, 50.0);
disp(s_amp.NumPorts);
disp(tsS21(s_amp));    % 10.0 + 0j (linear gain at both freqs)

% Passive with 3 dB insertion loss.
s_pas = rfAnalyzePassive(3.0, freqs, 50.0);
disp(tsS21(s_pas));    % 10^(-3/20) ≈ 0.7079 + 0j

% Series-R element (Z = 25 Ω resistive).
s_ser = rfAnalyzeSeries(25.0, 0.0, freqs, 50.0);
disp(tsS11(s_ser));    % 25/(25+100) = 0.2 + 0j
disp(tsS21(s_ser));    % 100/(25+100) = 0.8 + 0j

% Shunt-G element (Y = 0.01 S, R_shunt = 100 Ω).
s_sh = rfAnalyzeShunt(0.01, 0.0, freqs, 50.0);
disp(tsS11(s_sh));     % -(0.5)/(2.5) = -0.2 + 0j
disp(tsS21(s_sh));     % 2/2.5 = 0.8 + 0j
