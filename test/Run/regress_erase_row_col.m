% Regression: 2D row/column deletion A(i,:)=[] / A(:,j)=[] must shrink the
% array (MATLAB). Previously a no-op (the erase_rows/erase_cols runtime
% helpers existed but were never wired from lowering). Scalar and `end`
% indices are covered here (#189).
A = [1 2; 3 4; 5 6];
A(2,:) = [];                       % delete row 2 -> [1 2; 5 6]
fprintf('row: %.0fx%.0f vals=%.0f %.0f %.0f %.0f\n', ...
        size(A,1), size(A,2), A(1,1), A(1,2), A(2,1), A(2,2));
B = [1 2; 3 4];
B(:,1) = [];                       % delete col 1 -> [2; 4]
fprintf('col: %.0fx%.0f vals=%.0f %.0f\n', size(B,1), size(B,2), B(1), B(2));
C = [1 2; 3 4; 5 6];
C(end,:) = [];                     % delete last row -> [1 2; 3 4]
fprintf('end: %.0fx%.0f last=%.0f\n', size(C,1), size(C,2), C(2,2));
D = [10 20 30; 40 50 60];
k = 2;
D(:,k) = [];                       % variable index -> [10 30; 40 60]
fprintf('var: %.0fx%.0f vals=%.0f %.0f\n', size(D,1), size(D,2), D(1,2), D(2,2));
