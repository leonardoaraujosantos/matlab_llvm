% Exercises the trailing-comment scanner: `%` inside string literals
% must NOT be picked up as the start of a comment.
disp('50% done');     % trailing past a single-quoted string with %
disp("75% done");     % trailing past a double-quoted string with %
disp('hi');
