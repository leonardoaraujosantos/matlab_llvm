% A data-dependent while-loop is unsynthesizable in Phase 1 — needs
% explicit FSM extraction (Phase 4).
T = numerictype(1, 16, 0);
y = countdown(fi(5, T));
disp(y);

function y = countdown(x)
    y = x;
    while y > 0
        y = y - x;
    end
end
