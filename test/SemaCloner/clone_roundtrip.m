% Phase 2 of #38 — AST cloner round-trip gate.
%
% matlabc with -test-ast-clone deep-clones each top-level function with
% a `__clone` suffix and inserts the clones into the same TU before
% Sema runs. The golden asserts that every original function and its
% clone produce structurally identical Sema state (apart from their
% names). If the cloner forgets to zero out a Sema pointer or shares
% an AST sub-tree, the dump will diverge.

% Exercise a variety of node kinds so the cloner is tested broadly:
% binary ops, unary, postfix, range, calls, indexing, matrix literal,
% if-elseif-else, for, while.

function y = head(x)
    if x > 0
        y = x .* x;
    elseif x == 0
        y = 1;
    else
        y = -x;
    end
end

function y = sweep(n)
    y = 0;
    for k = 1:n
        y = y + k;
    end
    while y < 100
        y = y + 1;
    end
end

function y = mix(a, b)
    M = [a b; b a];
    y = M(1, 2) + a' - b;
end
