M = [1 2 3; 4 5 6; 7 8 9];
R = upper_left_2x2(M);

function S = upper_left_2x2(A)
    S = A(1:2, 1:2);
end
