% CommCRCGenerator — minimal parity-bit (degree-1) System Object.
%
% Proves the `obj(args)` -> `step(obj, args)` sugar + handle state-
% persistence across calls.  See `runtime/comm_class_crc_detector.m`
% for the companion detector and `comm_class_conv_encoder.m` /
% `comm_class_viterbi_decoder.m` for the convolutional code pair.
%
% MATLAB API: `comm.CRCGenerator('Polynomial', poly)` — flat name
% used here pending package-syntax support.

classdef CommCRCGenerator < handle
    properties
        % Generator polynomial marker (degree-1 today; full bit-pattern
        % in CommCRC8/CRC16/CRC32 follow-on classes).
        Polynomial
        % Running CRC remainder; persists across `step` calls.
        State
    end
    methods
        function obj = CommCRCGenerator(poly)
            if nargin == 1
                obj.Polynomial = poly;
            else
                obj.Polynomial = 1;
            end
            obj.State = 0;
        end

        function out = step(obj, bit)
            obj.State = mod(obj.State + bit, 2);
            out = obj.State;
        end

        function reset(obj)
            obj.State = 0;
        end
    end
end
