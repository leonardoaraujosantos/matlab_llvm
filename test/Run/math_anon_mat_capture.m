A = [1 2 3; 4 5 6; 7 8 9];
diagi = @(i) A(i, i);
disp(diagi(1));
disp(diagi(2));
disp(diagi(3));

M = [1 2; 3 4];
scaled = @(s) M * s;
disp(scaled(2));
