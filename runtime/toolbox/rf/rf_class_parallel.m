% RFCktParallel — parallel combination of RF circuit blocks.
%
% For two 2-port networks in parallel (shared input + output),
% the combined Y-matrix is the sum of the individual Y-matrices.  The
% v1 RFCktParallel skeleton holds per-block S-parameter column entries;
% population from RFCktPassive / RFCktAmplifier blocks via
% `parallel.append(block)` is a follow-on.

classdef RFCktParallel < handle
    properties
        NumBlocks
        S11_blocks      % cell-style placeholder for per-block S11
        S22_blocks      % per-block S22
        Frequency_Hz
        Label
    end
    methods
        function obj = RFCktParallel(num_blocks)
            if nargin >= 1
                obj.NumBlocks = num_blocks;
            else
                obj.NumBlocks = 0;
            end
            obj.Frequency_Hz = 1.0e9;
            obj.Label = "parallel";
        end
    end
end
