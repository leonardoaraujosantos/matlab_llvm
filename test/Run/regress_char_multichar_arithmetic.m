% regress_char_multichar_arithmetic.m — multi-char char-array arithmetic (#234).
% A multi-char single-quoted literal promotes to a 1xN row of its character
% codes in arithmetic: `'AB' + 1 == [66 67]`, `'abc' - 'a' == [0 1 2]`. (The
% single-char case `'A' + 1 == 66` and char-vs-number comparison live in
% regress_char_arithmetic.m.) Before the fix a multi-char CharLiteral was typed
% as a matlab_string* and left the matlab.add unconverted. TypeInference now
% types it as a 1xN double row, and the BinaryOp lowering materialises the codes
% via concat_row (the same path a numeric row literal `[65 66]` takes).
%
% Kept separate from regress_char_arithmetic.m because the matrix results print
% in numpy's `[[66. 67.]]` form under -emit-python / -emit-typescript, which
% can't match MATLAB's column-aligned layout — hence the .skip-emit-python /
% .skip-emit-typescript markers. The Run / emit-c / emit-cpp lanes (which use
% the MATLAB-format runtime disp) cover it.

disp('AB' + 1);       % 66 67
disp('AB' + 0);       % 65 66
disp('abc' - 'a');    % 0 1 2
disp(2 * 'AB');       % 130 132  (scalar * multi-char)
y = 'AB' + 1;
disp(y);              % 66 67   (assignment)
