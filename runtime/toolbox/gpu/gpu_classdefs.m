% gpu_classdefs.m — GPU Coder host-side carriers.
%
% On the CPU-debug lane `gpuArray` is an identity builtin
% (matlab_gpuArray_ctor returns its input) rather than a carrier object,
% so host matrix ops (mtimes / gather / size / …) operate directly on the
% wrapped matrix. See #333 and docs/gpu_coder_roadmap.md §1. This file is
% kept as the prelude target for the gpu names but defines no classdef.
