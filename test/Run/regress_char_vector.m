% Regression for #234 symptom 3: char([codes]) builds a string from a vector
% of integer code points (routes to the matlab_char_m runtime once the
% matrix-literal operand is materialised). The scalar char(code) form still
% works. char of a matrix variable is also exercised. disp prints the
% resulting string identically across all four execute backends.
disp(char([72 73]));
disp(char([72 101 108 108 111]));
disp(char(65));
v = [87 111 114 108 100];
disp(char(v));
