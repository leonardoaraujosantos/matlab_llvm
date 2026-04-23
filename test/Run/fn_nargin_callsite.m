% Same function called with different arities gets a per-arity clone
% so that nargin inside the body reflects the actual call-site count.
disp(add2(5, 7));      % nargin == 2
disp(add2(5));         % nargin == 1

function y = add2(a, b)
    if nargin == 2
        y = a + b;
    else
        y = a + 100;
    end
end
