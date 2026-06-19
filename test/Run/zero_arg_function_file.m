% Regression for #329: a .m file containing only a zero-argument function
% (no top-level script statements) runs like a script in MATLAB — by calling
% the function. matlabc synthesizes a main() that calls the primary function
% so the AOT lanes link + run instead of failing with "no main entry".
% Arg-taking function files stay no-main by design (run-tests skips them).
function zero_arg_function_file()
    x = 6;
    y = x * 7;
    disp(y)
    z = sum([1 2 3 4]);
    disp(z)
end
