% Phase-2 fi: implicit fi + double promotion. The double literal is cast
% to the fi's numerictype before the binop.
a = fi(1.5, 1, 16, 8);
b = fi(0, 1, 17, 8);
b(:) = a + 0.25;           % 0.25 quantizes to stored 64; sum 448; real-world 1.75
disp(b);

c = fi(0, 1, 17, 8);
c(:) = 0.5 + a;            % symmetric
disp(c);
