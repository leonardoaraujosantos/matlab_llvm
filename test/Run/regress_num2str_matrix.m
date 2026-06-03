% regress_num2str_matrix.m — regression test for num2str of a vector/matrix
% (#169). Before the fix only the scalar f64 form was wired; a vector/matrix
% argument failed with "unsupported call shape". A ptr operand now routes to
% matlab_num2str_mat, which "%g"-formats each element (two spaces between
% row elements, newline between rows).

% --- row vector ----------------------------------------------------
disp(num2str([1 2 3]));        % 1  2  3

% --- decimals ------------------------------------------------------
disp(num2str([1.5 2.25 3]));   % 1.5  2.25  3

% --- in a char concatenation (the common label idiom) --------------
disp(['v = ' num2str([10 20 30])]);   % v = 10  20  30

% --- scalar num2str still works ------------------------------------
disp(num2str(3.14));           % 3.14
disp(num2str(42));             % 42

% --- single-element vector behaves like scalar ---------------------
disp(num2str([7]));            % 7
