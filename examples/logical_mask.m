% Extract entries from a matrix via a logical mask.
A = [ 1 -2  3;
     -4  5 -6;
      7 -8  9];

disp('A =');
disp(A);

disp('positive entries (flattened):');
disp(A(A > 0));

disp('negative entries (flattened):');
disp(A(A < 0));

% Count entries above the mean. We use sum-of-logical (MATLAB's
% idiomatic counting trick) rather than a fprintf of the count so the
% demo stays inside types Sema has inferred.
disp('mean =');
disp(mean(A(:)));

disp('# entries strictly above mean:');
disp(sum(A(:) > mean(A(:))));
