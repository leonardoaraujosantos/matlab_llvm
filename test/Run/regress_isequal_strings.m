% regress_isequal_strings.m — regression test for isequal on string
% operands (#147). Before the fix, matlab_isequal took two matlab_mat* and
% read rows/cols/data; a matlab_string has a different layout, so
% isequal("ab","ab") mis-read the string and returned 0. The frontend now
% routes a both-string isequal to the strcmp path (#146), which does a
% length + byte compare. Non-string isequal still uses matlab_isequal.

% --- equal / not-equal strings -------------------------------------
disp(isequal("ab", "ab"));        % 1   (was 0)
disp(isequal("hello", "hello"));  % 1
disp(isequal("ab", "xy"));        % 0
disp(isequal("ab", "abc"));       % 0  (length differs)

% --- string variables ----------------------------------------------
a = "match";
b = "match";
disp(isequal(a, b));              % 1
c = "other";
disp(isequal(a, c));              % 0

% --- numeric isequal is unaffected ---------------------------------
disp(isequal([1 2 3], [1 2 3]));  % 1
disp(isequal([1 2 3], [1 2 4]));  % 0
disp(isequal([1 2], [1 2 3]));    % 0  (shape differs)

% --- result usable in a guarded condition --------------------------
if isequal("yes", "yes") == 1
  disp(42);                       % 42
end
