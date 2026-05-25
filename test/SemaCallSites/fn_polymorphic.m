% Phase 1 of the Sema-time monomorphization epic (#38).
%
% Gates the CallSiteAnalyzer's per-callee signature bucketing. The
% expected golden asserts that `sq` and `twice` each get exactly the
% set of signatures their call sites use — scalar and matrix shapes
% become distinct buckets. Phase 3 of the epic will consume this
% analysis to clone helpers per signature.

disp(sq(5));
disp(sq([1 2 3]));
disp(twice([1 2; 3 4]));
disp(twice(7));

function y = sq(x)
    y = x .* x;
end

function y = twice(x)
    y = x + x;
end
