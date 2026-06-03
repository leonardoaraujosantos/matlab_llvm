% regress_strcmp.m — regression test for the strcmp / strcmpi builtins
% (#146). Before the fix these were `undefined name` (Sema rejected the
% call), so any program using them failed to compile. strcmp returns 1.0
% when the strings are equal, 0.0 otherwise (MATLAB sense — note this is
% the opposite of C's strcmp). strcmpi is the case-insensitive form.
%
% Uses the supported call shapes (double-quoted string args; result used
% as a value or in an explicit `== 1` comparison). The direct
% `if strcmp(a,b)` form and single-quoted char-array args share a
% pre-existing limitation of the whole string-predicate family
% (contains/startsWith/...), tracked separately.

% --- equal / not-equal ---------------------------------------------
disp(strcmp("abc", "abc"));     % 1
disp(strcmp("abc", "abd"));     % 0

% --- different lengths are not equal -------------------------------
disp(strcmp("ab", "abc"));      % 0

% --- case sensitivity ----------------------------------------------
disp(strcmp("ABC", "abc"));     % 0  (case-sensitive)
disp(strcmpi("ABC", "abc"));    % 1  (case-insensitive)
disp(strcmpi("MixedCase", "mixedcase"));  % 1

% --- result usable as a value / in a guarded condition -------------
eq = strcmp("hi", "hi");
disp(eq + 10);                  % 11

if strcmp("yes", "yes") == 1
  disp(42);                     % 42
end
