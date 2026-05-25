% Phase 1 wider gate (#38). Mirrors test/Run/fn_polymorphic_invariant.m
% but here we assert on the analyzer's per-signature buckets rather
% than on the runtime behaviour. Multi-arg helpers, every shape category
% (scalar / row / col / matrix), and complex must each surface as a
% distinct bucket — Phase 3 cloning depends on this fidelity.

% --- 1. Polymorphic add — scalar / row vector / column vector / matrix
disp(addtwo(2.5,  -0.5));
disp(addtwo([1 2 3], [10 20 30]));
disp(addtwo([1; 2], [10; 20]));
disp(addtwo([1 2; 3 4], [10 20; 30 40]));

% --- 2. Polymorphic single-arg helper — scalar / row / matrix / complex
disp(square(4));
disp(square([1 2 3]));
disp(square([1 2; 3 4]));
disp(square(1 + 2i));

function y = addtwo(a, b)
    y = a + b;
end

function y = square(x)
    y = x .* x;
end
