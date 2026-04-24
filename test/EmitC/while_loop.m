% Exercises the scf.while emission shape. Target: `while (i < 10) { ... }`
% rather than the `while (1) { if (!cond) break; ... }` intermediate form.
i = 0;
while i < 10
    disp(i);
    i = i + 1;
end
