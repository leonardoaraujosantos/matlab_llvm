% Issue #20 follow-on — outer-scope slots that are purely iteration-local
% inside the parfor body (e.g. `cim = im_min + ...; total = total + cim;`
% where `cim` is written-then-read every iteration).  The MATLAB frontend
% allocates these as outer matlab.alloc ops even though they're local
% to the body; OutlineParfor now clones the alloc into the outlined
% function entry so each iteration has its own private slot.
%
% Also covers iteration-local slots introduced by NESTED for/while loops
% inside the parfor (their IV slots + intermediate locals), which the
% direct-block scan would miss — the detection walks the parfor region
% transitively.

% Plain iteration-local intermediate.
re_min = -2.0;
re_range = 3.0;
N = 60;
total = 0;
parfor py = 1:N
    cre = re_min + re_range * (py - 1) / (N - 1);
    total = total + cre;
end
% Sum of linspace(-2, 1, 60) is -30.0.
fprintf('cre_sum = %.1f\n', total);

% Iteration-local intermediate inside a nested for loop.
W = 10;
acc = 0;
parfor py = 1:5
    row_count = 0;
    for px = 1:W
        row_count = row_count + py * px;
    end
    acc = acc + row_count;
end
% sum_{py=1..5} sum_{px=1..10} py*px = (1+2+..+10) * (1+..+5) = 55 * 15 = 825.
fprintf('acc = %d\n', acc);
