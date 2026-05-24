% gpu_config_classdefs.m — coder.gpuConfig carrier (separate file from
% gpuArray to keep the AOT-prelude path happy with single-classdef files).

classdef coder_gpuConfig < handle
    properties
        Target
        EnableCUBLAS
        EnableCUSOLVER
        EnableCUFFT
        EnableMemoryManager
        EnableMPS
        StackLimitPerThread
        HalfType
        OpenCLPlatform
    end
    methods
        function obj = coder_gpuConfig(target)
            obj.Target = 1;
            obj.EnableCUBLAS = 1;
            obj.EnableCUSOLVER = 1;
            obj.EnableCUFFT = 1;
            obj.EnableMemoryManager = 1;
            obj.EnableMPS = 1;
            obj.StackLimitPerThread = 1024;
            obj.HalfType = 0;
            obj.OpenCLPlatform = 0;
        end
    end
end
