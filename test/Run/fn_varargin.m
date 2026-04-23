% Variadic user function: last declared param named varargin receives
% a cell of the trailing call-site args.
disp(sumall(1));
disp(sumall(1, 2, 3));
disp(sumall(10, 20, 30, 40, 50));

% varargin interacts with nargin — each distinct call-site arity gets
% its own clone with the right compile-time nargin value.
disp(report(100, 1));
disp(report(100, 1, 2, 3));

function y = sumall(a, varargin)
    y = a;
    n = numel(varargin);
    i = 1;
    while i <= n
        y = y + varargin{i};
        i = i + 1;
    end
end

function y = report(base, varargin)
    y = base + nargin;
end
