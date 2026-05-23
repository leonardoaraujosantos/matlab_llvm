% fprintf: >4 trailing values (arity) + %d / %i on double args (integer print).
fprintf('%d %d %d %d %d %d\n', 1, 2, 3, 4, 5, 6);
fprintf('%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n', 1, 2, 3, 4, 5, 6, 7, 8);
x = [10 20 30];
fprintf('count=%d sum=%d\n', numel(x), sum(x));
fprintf('%5d|%-3d|%i\n', 7, 8, 9);
fprintf('%d%% of %.2f\n', 50, 3.14159);
