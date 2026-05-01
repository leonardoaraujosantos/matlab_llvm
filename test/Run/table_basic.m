% Phase 5.3 — table: column-major record with named variables.
% Constructors (auto-named + 'VariableNames'), column access via dot,
% column write, height / width / size / disp, dynamic column add.

% Auto-named columns get Var1, Var2, ...
T = table([1; 2; 3; 4], [10; 20; 30; 40]);
disp(height(T));     % 4
disp(width(T));      % 2
disp(T);

v = T.Var1;
disp(v);             % column 1

% Replace existing column.
T.Var2 = [100; 200; 300; 400];
disp(T.Var2);

% Explicit VariableNames.
U = table([1; 2; 3], [4.5; 5.5; 6.5], 'VariableNames', {'id', 'score'});
disp(U);
disp(U.id);
disp(U.score);
disp(height(U));
disp(width(U));

% Dynamic column add.
U.bonus = [10; 20; 30];
disp(U);
disp(width(U));
