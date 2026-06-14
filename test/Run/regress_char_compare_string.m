% regress_char_compare_string.m — regression test for #265: a char-array
% VARIABLE used in a comparison or arithmetic operates element-wise on the
% character CODES (like MATLAB), not as a string. Before the fix a char
% variable (`c = 'hello'`) lowered to a non-materialized char value
% (tensor<1xNxi8> in AOT, matlab_string* in the REPL), so `c == 'l'` left
% matlab.eq unconverted ("unsupported call shape") and `c + 1` wrongly took
% the string-concat path. The AOT lane now materializes the const_char into
% an f64 code matrix in LowerTensorOps; the concat path only fires for a
% genuine `string` operand, so char `+`/`==`/`<` fall through to the numeric
% matrix runtime. (Char LITERAL arithmetic is covered by
% regress_char_arithmetic.m; this exercises the char-VARIABLE path.)

c = 'hello';

% --- comparison operators on char-array codes ----------------------
disp(c == 'l');       % 0 0 1 1 0
disp(c ~= 'l');       % 1 1 0 0 1
disp(c < 'l');        % 1 1 0 0 0
disp(c >= 'l');       % 0 0 1 1 1

% --- arithmetic on char-array codes --------------------------------
disp(c + 1);          % 105 102 109 109 112
disp(c - 32);         % 72 69 76 76 79  (upper-case codes)

% --- char LITERAL comparison still works ---------------------------
disp('hello' == 'l'); % 0 0 1 1 0

% --- a genuine string ("...") still concatenates -------------------
s = "hi";
disp(s + "!");        % hi!
