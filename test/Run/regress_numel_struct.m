% regress_numel_struct.m — regression test for numel/length of a struct
% (#177). Before the fix, numel(s)/length(s) read the struct pointer as a
% matlab_mat and returned garbage; a scalar struct now yields 1 (the
% numel/length lowering gained a struct case alongside the cell one).

% --- scalar struct -------------------------------------------------
s.a = 1;
s.b = 2;
disp(numel(s));     % 1
disp(length(s));    % 1

% --- nested struct (still one element) -----------------------------
t.x.y = 3;
disp(numel(t));     % 1

% --- usable in a guard ---------------------------------------------
if numel(s) == 1
  disp(42);         % 42
end

% --- cell / matrix numel unaffected --------------------------------
c = {1, 2, 3};
disp(numel(c));     % 3
disp(numel([1 2 3; 4 5 6]));   % 6
disp(length([1 2 3 4]));       % 4
