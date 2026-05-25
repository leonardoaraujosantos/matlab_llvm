% Phase 3 acceptance gate for #38 — specialization driver.
%
% After the monomorphizer runs to fixpoint, the AST has 4 function
% definitions (`sq`, `sq__s1`, `twice`, `twice__s1`) and each call
% site at the script level resolves to exactly one. The canonical
% signature for each callee (alphabetically first) keeps the original
% name; the others get `__sN` suffixes.

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
