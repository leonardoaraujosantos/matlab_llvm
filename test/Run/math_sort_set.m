% Sort / unique / set operations.

v = [3 1 4 1 5 9 2 6 5 3 5];
disp(sort(v));               % [1 1 2 3 3 4 5 5 5 6 9]
disp(unique(v));             % column: [1; 2; 3; 4; 5; 6; 9]

A = [3 1; 1 2; 2 3];
disp(sort(A));               % column-wise: [1 1; 2 2; 3 3]

disp(sortrows(A));           % lex: [1 2; 2 3; 3 1]

a = [1 2 3 4];
b = [3 4 5 6];
disp(intersect(a, b));       % [3; 4]
disp(setdiff(a, b));         % [1; 2]
disp(union(a, b));           % [1; 2; 3; 4; 5; 6]

disp(ismember(a, b));        % [0 0 1 1]
