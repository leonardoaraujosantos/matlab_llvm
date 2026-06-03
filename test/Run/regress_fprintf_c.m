% Regression: fprintf %c must emit the character for a numeric value (MATLAB),
% not print the number. Previously %c was grouped with %s and printed %g. (#209)
fprintf('%c\n', 65);             % A
fprintf('%c%c%c\n', 72, 73, 33); % HI!
fprintf('[%c]=%d\n', 65, 1);     % [A]=1
fprintf('%c', [72 73 74]);       % HIJ (format recycles over the vector)
fprintf('\n');
fprintf('%s and %c\n', "str", 88);  % str and X  (%s unaffected)
