% Phase-5 fi: all five rounding modes for the constructor quantize path.
% Q-format Q16.0 with the test values picked to exhibit the different
% behaviours at exact halves and on negative inputs.
T = numerictype(1, 16, 0);
disp(fi(2.5,  T, fimath('RoundingMethod','Floor')));      % 2
disp(fi(2.5,  T, fimath('RoundingMethod','Nearest')));    % 3
disp(fi(2.5,  T, fimath('RoundingMethod','Zero')));       % 2
disp(fi(2.5,  T, fimath('RoundingMethod','Ceiling')));    % 3
disp(fi(2.5,  T, fimath('RoundingMethod','Convergent'))); % 2 (round to even)
disp(fi(3.5,  T, fimath('RoundingMethod','Convergent'))); % 4 (round to even)
disp(fi(-2.5, T, fimath('RoundingMethod','Floor')));      % -3
disp(fi(-2.5, T, fimath('RoundingMethod','Zero')));       % -2
disp(fi(-2.5, T, fimath('RoundingMethod','Ceiling')));    % -2
