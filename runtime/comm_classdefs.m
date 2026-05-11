% Communications Toolbox System Object classdefs.
%
% Auto-prepended by matlabc when the user input mentions one of the
% CommCRC* / CommViterbi* / CommOFDM* names as a call target or LHS
% — see `findCommPrelude` and `userMentionsCommClasses` in
% `tools/matlabc/main.cpp`.
%
% MATLAB's standard surface is `comm.CRCGenerator` (dot-namespaced
% packages).  Until matlabc handles package syntax we ship the
% flat-name aliases `CommCRCGenerator`, `CommViterbiDecoder`, …;
% MathWorks tutorial code that imports `comm.*` can be ported by
% rewriting `comm.CRCGenerator` to `CommCRCGenerator` mechanically.
%
% Surface today:
%   CommCRCGenerator — minimal parity-bit (degree-1) System Object,
%       proves the `obj(args)` → `step(obj, args)` sugar + handle
%       state-persistence across calls.  Subsequent CRC widths (CRC-8
%       ATM HEC, CRC-16 CCITT, CRC-32) layer the same shape on top.

classdef CommCRCGenerator < handle
    properties
        % `Polynomial` is the generator polynomial; for the parity-bit
        % default it's just the marker `1`.  Subsequent CRC widths
        % carry the bit-pattern (MSB-first binary vector).
        Polynomial
        % `State` is the current CRC remainder.  Persists across
        % `step` calls — feeding one byte at a time and reading
        % `obj.State` after each call mirrors hardware shift-register
        % behavior, which is what the System Object idiom buys.
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
            % Parity update: XOR the new bit into the running parity.
            % For the degree-1 polynomial that's the entire CRC.  The
            % `mod(..., 2)` keeps the result in {0, 1} without leaning
            % on `bitxor` (Sema treats the scalars as f64 so plain
            % addition + mod is the cleanest path).
            obj.State = mod(obj.State + bit, 2);
            out = obj.State;
        end

        function reset(obj)
            obj.State = 0;
        end
    end
end
