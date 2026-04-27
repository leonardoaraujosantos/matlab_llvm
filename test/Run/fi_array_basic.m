% Phase-3 fi: array zero-init, length, numel, size, indexing.
a = fi(zeros(1, 5), 1, 16, 8);
disp(length(a));
disp(numel(a));
disp(size(a, 2));
disp(a(1));     % zero
disp(a(5));     % zero
