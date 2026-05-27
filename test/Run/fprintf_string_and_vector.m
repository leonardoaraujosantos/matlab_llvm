% fprintf %s (string variable + char literal) and vector-argument expansion
% (format recycling, multi-spec fill, column-major matrix order).
s = "world";
fprintf('hello %s\n', s);
fprintf('%s!\n', 'done');

% A single format spec recycles over every element of the vector.
fprintf('%d ', [1 2 3 4 5]);
fprintf('\n');

% A vector fills multiple specifiers in one pass.
fprintf('%d-%d-%d\n', [7 8 9]);

% Mixed string + numeric arguments.
n = "iter";
fprintf('%s %d of %d\n', n, 3, 10);

% Matrix elements are consumed column-major (MATLAB order).
fprintf('%g ', [1 2; 3 4]);
fprintf('\n');

% Float formatting recycles too.
fprintf('%.1f ', [1.5 2.5 3.5]);
fprintf('\n');

% sprintf shares the same core: %s, vector expansion, mixed args (string fmt).
disp(sprintf("[%s]", "hi"));
disp(sprintf("%d-", [1 2 3]));
disp(sprintf("%s=%.1f", "k", 2.5));

% fprintf(fid, ...) to a file routes through the same variadic core.
fid = fopen("/tmp/matlab_fprintf_vec_test.txt", "w");
fprintf(fid, "%d ", [10 20 30]);
fprintf(fid, "%s\n", "end");
fclose(fid);
rid = fopen("/tmp/matlab_fprintf_vec_test.txt", "r");
disp(fgetl(rid));
fclose(rid);
