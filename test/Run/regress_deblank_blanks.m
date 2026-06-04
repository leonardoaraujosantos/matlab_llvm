% deblank(s): strip only trailing whitespace (leading preserved) — distinct
% from strtrim. blanks(n): an n-space string. Both were unrecognised.
fprintf('[%s]\n', deblank('hi   '));
fprintf('[%s]\n', deblank('  keep   '));
fprintf('[%s]\n', blanks(3));
s = 'world  ';
fprintf('[%s]\n', deblank(s));
fprintf('[%s|%s]\n', blanks(0), strcat(blanks(2), 'x'));
