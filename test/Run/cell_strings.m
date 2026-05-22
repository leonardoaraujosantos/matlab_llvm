% Cell-of-strings literal passed to legend. This was a compile error
% (`matlab_cell_set_mat: 3 arguments` on the char-tensor element); a naive
% "store as matlab_string via set_mat" fix then made legend bad_alloc.
% String elements now store with kind=3 (matlab_cell_set_str) and
% matlab_cell_get_mat exposes them as char-code rows, so legend reads them.
x = 1:5;
plot(x, x);
hold on;
plot(x, x .^ 2);
legend({'linear', 'quadratic'});
title('cell-of-strings legend');
disp('legend ok');
