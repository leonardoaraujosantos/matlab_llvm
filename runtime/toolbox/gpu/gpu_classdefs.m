% gpu_classdefs.m — GPU Coder host-side carriers.
%
% Auto-prepended by matlabc when the user source mentions any of:
%   gpuArray, gather, existsOnGPU, gpuDevice
%
% (coder.gpuConfig lives in a separate file — gpu_config_classdefs.m
% — because the parser AOT path only consistently accepts a single
% classdef per prelude file when prepended to a function-defining
% user input; splitting umbrellas keeps the test suite green.)
%
% T1 design: gpuArray is a host-only handle carrier — it stores the
% Underlying matrix value and metadata (Device target, dtype).  The
% CPU-debug lane satisfies every read by returning Underlying directly.

classdef gpuArray < handle
    properties
        Underlying
        Device
        Dtype
        DevicePtr
    end
    methods
        function obj = gpuArray(x)
            obj.Underlying = x;
            obj.Device = 0;
            obj.Dtype = 0;
            obj.DevicePtr = 0;
        end
    end
end
