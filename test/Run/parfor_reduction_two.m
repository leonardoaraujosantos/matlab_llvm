a = 0;
b = 0;
parfor i = 1:5
    a = a + i;
    b = b + 2 * i;
end
disp(a);
disp(b);
