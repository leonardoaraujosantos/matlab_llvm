% Loop-carried vector with element indexing in arithmetic.
v = [1.0; 1.0; 1.0; 1.0];
for n = 1:15
    s = v(1) + v(2) + v(3) + v(4);
    v = v + 0.01 * [s; -s; s; -s];
end
disp(v);
