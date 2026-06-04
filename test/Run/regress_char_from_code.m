% char(code): build a 1-char string from a numeric code point. Was fully
% unsupported ("unsupported call shape"); now lowers to matlab_char_s -> a
% matlab_string*, so it disp's/concatenates as text. (char of a vector and
% char-as-numeric-array arithmetic remain separate follow-ups.)
fprintf('%s %s %s\n', char(65), char(97), char(72));
c = char(66);
fprintf('%s\n', strcat(c, 'CD'));
k = 90;
fprintf('%s\n', char(k));
