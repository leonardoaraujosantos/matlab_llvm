% Phase-4 fi: numerictype object as fi() second argument.
T = numerictype(1, 16, 8);
a = fi(1.5, T);
b = fi(2.25, T);
c = fi(0, 1, 17, 8);
c(:) = a + b;
disp(c);
disp(a);
disp(b);
