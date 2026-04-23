C = {1, 2, 3, 4};
disp(numel(C));
disp(length(C));
disp(iscell(C));

x = 42;
disp(iscell(x));

% Auto-grow reflected in numel.
C{7} = 99;
disp(numel(C));
