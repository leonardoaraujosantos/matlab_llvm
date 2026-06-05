% Regression for #234 (minimum safe step): length / numel of a runtime
% string produced inline by a string-returning builtin must return the char
% count via matlab_string_len, instead of reading the matlab_string
% descriptor as a matrix (which produced a garbage double / UB). The
% char-literal length (const_char fold) is unchanged. Printed via
% fprintf %.0f so output is byte-identical across all four execute backends.
fprintf('%.0f %.0f\n', length(blanks(5)), numel(blanks(7)));
fprintf('%.0f %.0f\n', length(strcat('ab','cd')), numel(strcat('x','yz')));
fprintf('%.0f %.0f\n', length(strtrim('  hi  ')), length(upper('abc')));
fprintf('%.0f\n', length('hello'));
