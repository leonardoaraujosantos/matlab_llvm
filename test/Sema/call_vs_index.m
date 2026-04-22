function r = demo()
    x = [10 20 30];
    a = x(2);       % Index (x is a variable)
    b = sin(0.5);   % Call  (sin is a builtin)
    c = zeros(3);   % Call  (zeros is a builtin)
    r = a + b + c(1,1);
end
