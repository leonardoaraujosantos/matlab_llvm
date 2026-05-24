% Issue #21 tightening proposal — invariant gate for the
% PromoteNoneParams in-module-caller skip workaround.
%
% Today PromoteNoneParams skips any user function with an in-module
% caller, leaving its `none` arg untouched so the late monomorphiser
% (runMonomorphiseUserCalls inside LowerUserCalls) can clone the body
% per concrete signature seen at call sites.
%
% Without the workaround, PromoteNoneParams would settle the helper's
% arg type from body usage (numeric => f64) and the matrix call sites
% would fail to dispatch.  The existing fn_polymorphic.m exercises
% sq(scalar) + sq(vector); this fixture WIDENS the surface so that if
% the workaround is ever pre-empted or sidestepped, multiple call-site
% combinations break at once instead of one slipping through.
%
% Per #21's "Tightening proposal": "Add a regression test that asserts
% the *invariant the workaround maintains* (not just the visible
% behavior)".  Each shape below would be the first to break if the
% in-module-caller guard stopped firing — so this is the canary.

% --- 1. Polymorphic add — scalar / row vector / column vector / matrix
disp(addtwo(2.5,  -0.5));            % scalar + scalar -> scalar
disp(addtwo([1 2 3], [10 20 30]));    % row + row -> row
disp(addtwo([1; 2], [10; 20]));       % col + col -> col
disp(addtwo([1 2; 3 4], [10 20; 30 40])); % mat + mat -> mat

% --- 2. Polymorphic single-arg helper — scalar / row vector / matrix /
%     complex scalar.  Mixing complex into the same helper is the
%     surface most at risk if param-type settles to plain f64.
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
