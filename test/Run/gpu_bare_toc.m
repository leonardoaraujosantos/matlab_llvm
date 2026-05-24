% Bare-name `toc` / `gpuDeviceCount` / `gpuDevice` calls — MATLAB
% implicit-call syntax (no parens).  Without this fix, `gpuTime = toc;`
% would emit matlab.make_handle (a function reference, not a call).
tic;
x = 1 + 2;
gpuTime = toc;
disp(gpuTime >= 0);
n = gpuDeviceCount;
disp(n);
h = gpuDevice;
wait(h);
disp('ok');
