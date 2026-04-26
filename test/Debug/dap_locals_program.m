% Fixture for the function-frame Locals + evaluate scenarios. The
% script calls compute() which has its own locals (a, b, sum). When
% paused inside compute(), the DAP `variables` response for the
% function frame must show the function's locals — not the script's
% workspace. Line numbers are referenced from dap_scenarios.py.
seed = 7;
result = compute(3, 4);
disp(result);

function s = compute(a, b)
    sum = a + b;
    s = sum * 2;
end
