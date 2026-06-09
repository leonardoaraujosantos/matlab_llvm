% regress_char_arithmetic.m — regression test for char-literal arithmetic
% and char-vs-number comparison (#234). A single-char literal promotes to its
% numeric code in ARITHMETIC (`'A' + 1 == 66`) and in a comparison whose other
% operand is numeric (`'A' == 65`), unlike a string literal ("A") which
% concatenates. Before the fix a CharLiteral operand was typed/lowered as a
% matlab_string*, so the matlab.add/eq op was left unconverted. visitBinary now
% re-types a single-char literal operand as a scalar double for numeric
% operators (always) and for comparisons (only when the OTHER operand is
% numeric, so `strvar == 'x'` keeps its string semantics); the BinaryOp
% lowering emits the char's code. "A" (StringLiteral) is left alone.

% --- char literal in arithmetic -> numeric code --------------------
disp('A' + 1);        % 66
disp('a' - 'A');      % 32  (case-shift distance)
disp('Z' + 0);        % 90
disp('0' + 5);        % 53  ('0' is 48)
disp(2 * '0');        % 96  (other operand order)
x = 'A' + 1;
disp(x);              % 66

% --- char vs number comparison -> compare codes --------------------
disp('A' == 65);      % 1
disp('A' == 66);      % 0
disp('Z' >= 65);      % 1
disp(65 == 'A');      % 1   (number on the left)

% --- a string literal ("...") still concatenates (not numeric) -----
disp("hi" + 5);       % hi5

% --- plain char display is unchanged -------------------------------
disp('A');            % A
