% [r, c] = size(A): multi-return destructuring. Used to crash
% silently because the Lowerer allocated tensor slots for the
% LHS names while the call already returns f64 scalars.
A = [1 2 3; 4 5 6];
[r, c] = size(A);
disp(r);
disp(c);
[m, n] = size([1; 2; 3; 4]);
disp(m);
disp(n);
