% A double-typed input has no synthesizable form without an explicit
% fi(...) conversion. Phase 1 rejects floating-point at the boundary.
y = scale(3.5, 2.0);
disp(y);

function y = scale(a, b)
    y = a * b;
end
