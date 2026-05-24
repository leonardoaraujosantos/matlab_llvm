% arrayfun with a NAMED function handle (@activation_kernel) instead of
% an anon @(...) — validates the LowerAnonCalls resolution of make_handle
% for user-defined func.func ops via func.constant + unrealized_conversion_cast.
y = run(5);
disp(y(3));

function y = run(n)
    x = gpuArray.linspace(-1.0, 1.0, n);
    y = gather(arrayfun(@activation_kernel, x));
end

function v = activation_kernel(x)
    v = 1.0 / (1.0 + exp(-x));
end
