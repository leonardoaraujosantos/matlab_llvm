% System Object MVP — `CommCRCGenerator` (parity-bit).
%
% Exercises:
%   - Handle classdef with persistent property mutation across calls.
%   - The `obj(args)` -> `step(obj, args)` syntactic sugar (System Object
%     callable-instance idiom).
%   - Auto-prepended `runtime/comm_classdefs.m` prelude (matlabc detects
%     `CommCRCGenerator(...)` as a call target and pulls the file in).
%   - Dot-method dispatch on a lifecycle method (`crc.reset()`).
%
% The classdef body lives in `runtime/comm_classdefs.m`; this script
% only references the type by name.

crc = CommCRCGenerator(1);
disp(crc(1));     % 1 — parity of [1]
disp(crc(0));     % 1 — parity of [1 0]
disp(crc(1));     % 0 — parity of [1 0 1]
disp(crc(1));     % 1 — parity of [1 0 1 1]
disp(crc(1));     % 0 — parity of [1 0 1 1 1]
crc.reset();
disp(crc(1));     % 1 — fresh start after reset
