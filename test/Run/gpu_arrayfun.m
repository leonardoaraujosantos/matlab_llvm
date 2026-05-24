% gpuArray PCT-surface arrayfun validation.  Exercises:
%   - gpuArray.linspace(start, stop, n)  — 3-arg numeric factory
%   - arrayfun(@anon, gpuArr)             — scalar kernel via function-handle ABI
%   - gather(g)
% On the CPU lane the anon is invoked sequentially per element;
% Tier-2/3/4 backends emit it as a per-thread kernel.
x = gpuArray.linspace(-1.0, 1.0, 5);
square = @(v) v * v;
y_gpu = arrayfun(square, x);
y = gather(y_gpu);
% Expect 1, 0.25, 0, 0.25, 1
disp(y);
