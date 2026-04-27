% Phase-4 fi: setfimath / removefimath swap the fimath without
% touching the underlying stored integer.
T = numerictype(1, 8, 0);
F_sat = fimath('OverflowAction', 'Saturate');
F_wrap = fimath('OverflowAction', 'Wrap');
a = fi(127, T, F_sat);
disp(a);                  % 127
b = setfimath(a, F_wrap); % stored int unchanged
disp(b);                  % 127
c = removefimath(b);      % defaults: Saturate / Floor
disp(c);                  % 127
