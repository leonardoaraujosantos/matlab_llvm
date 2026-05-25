% Phase 4 acceptance gate for #38 — recursion closure.
%
% A polymorphic helper that recurses into itself. Two outer call sites
% seed the analyzer with two concrete signatures (scalar + vector).
% The inner recursive call's args are typed `any` (TypeInference does
% not refine across user-function returns), so it surfaces as a third
% signature bucket — the canonical one that keeps the original name.
%
% Phase 4's fixpoint loop must:
%   - converge in bounded iterations (does not infinite-loop on the
%     recursive call site),
%   - produce two clones (`accum__s1`, `accum__s2`) for the concrete
%     signatures, with the canonical `accum` retained for the inner
%     recursive call.

disp(accum(0.0, 3));         % scalar entry  → accum__s1 (or canonical)
disp(accum([0 0 0], 3));     % vector entry  → accum__s2

function s = accum(s, n)
    if n > 0
        s = s + 1;
        s = accum(s, n-1);
    end
end
