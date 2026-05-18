% CommCRCDetector — companion to CommCRCGenerator.  Wraps the
% function-form `crcCheck` + `crcStrip` runtime helpers.
%
% Tracks total errors seen in `ErrorCount` (MATLAB's `comm.CRCDetector`
% surfaces this via the second return of step; v1 here uses a
% property + reset(obj) instead, since multi-return from a method is
% a follow-on).
%
% Polynomial encoding: decimal integer (same as `crcGenerate`):
%   CRC-8 ATM HEC  : 0x07 = 7
%   CRC-16 CCITT   : 0x1021 = 4129
%   CRC-32         : 0x04C11DB7 = 79764919

classdef CommCRCDetector < handle
    properties
        Polynomial
        NumBits
        ErrorCount
    end
    methods
        function obj = CommCRCDetector(poly, nbits)
            obj.Polynomial = poly;
            obj.NumBits = nbits;
            obj.ErrorCount = 0;
        end

        function msg = step(obj, codeword)
            err = crcCheck(codeword, obj.Polynomial, obj.NumBits);
            obj.ErrorCount = obj.ErrorCount + err;
            msg = crcStrip(codeword, obj.NumBits);
        end

        function reset(obj)
            obj.ErrorCount = 0;
        end
    end
end
