% gpuArray PCT-surface device-info validation.  Exercises:
%   - gpuDeviceCount()      — returns 1 on CPU lane
%   - gpuDevice(id)         — select by ID, returns ID
%   - existsOnGPU(g)        — returns 1 for any gpuArray
%   - wait(gpuDevice)       — synchronise (no-op on CPU lane)
fprintf('gpuDeviceCount = %.0f\n', gpuDeviceCount());
fprintf('gpuDevice(1)   = %.0f\n', gpuDevice(1));
x = gpuArray([1 2 3]);
fprintf('existsOnGPU    = %.0f\n', existsOnGPU(x));
h = gpuDevice();
wait(h);
disp('wait ok');
