% regress_cell_element_assign.m — #292: assigning a matrix / string / scalar
% into a cell element, plus cell(m,n) preallocation. Before the fix a dynamic
% `c{i} = matrix` read back as 0 (the read defaulted to matlab_cell_get_f64),
% `c{i} = 'str'` was an unsupported call shape (i8 const_char into
% matlab_cell_set_mat), and `cell(1,N)` preallocation didn't lower. The store
% now updates the binding's element-kind tracking so the brace read picks
% get_mat / get_str, char RHS is wrapped to a matlab_string*, and cell(...)
% lowers to matlab_cell_new[_2d].

% --- preallocate, then fill with mixed kinds -----------------------
c = cell(1, 3);
c{1} = [10 20; 30 40];   % matrix element
c{2} = 'hello';          % char/string element
c{3} = 42;               % scalar element
disp(c{1});
disp(c{2});
disp(c{3});
disp(numel(c));

% --- dynamic assign into a cell LITERAL ----------------------------
d = {1, 2};
d{1} = [7 8 9];
disp(d{1});
d{2} = 'world';
disp(d{2});

% --- subscript a cell-element result (already worked; lock it) ------
e = {[100 200 300]};
disp(e{1}(2));
