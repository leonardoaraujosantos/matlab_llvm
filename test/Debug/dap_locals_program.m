% Fixture for function-frame Locals + evaluate. Local names avoid
% builtins (sum / prod / ...) so the REPL JIT resolves them as ws
% vars rather than function refs. Line numbers referenced by
% dap_scenarios.py.
seed = 7;
result = compute(3, 4);
disp(result);

function s = compute(a, b)
    total = a + b;
    s = total * 2;
end
