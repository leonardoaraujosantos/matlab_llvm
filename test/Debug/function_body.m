% Script with a helper function — verifies hooks are injected inside
% the user function body too, not just at script scope.
y = add_one(41);
disp(y);

function r = add_one(x)
    % function-internal comment
    r = x + 1;
end
