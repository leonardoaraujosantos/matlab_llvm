% #233: strsplit returns a cell of strings. Exercises the cell-result slot
% typing (parts is ptr-typed), brace-read routing to matlab_cell_get_str for a
% constant AND a runtime index, fprintf %s of a cell string element, and the
% whitespace-default (collapsing) 1-arg form.
parts = strsplit('a,b,c', ',');
fprintf('n %d\n', numel(parts));
disp(parts{2});
k = 3;
disp(parts{k});
fprintf('s %s\n', parts{1});
w = strsplit('x y  z');
fprintf('wn %d\n', numel(w));
disp(w{2});
