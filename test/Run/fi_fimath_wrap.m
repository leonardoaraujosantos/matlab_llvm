% Phase-4 fi: fimath('OverflowAction','Wrap') is now reachable from
% MATLAB syntax (Phase 1 implemented Wrap in the runtime but only
% Saturate was settable).
T = numerictype(1, 8, 0);
F_wrap = fimath('OverflowAction', 'Wrap');
F_sat  = fimath('OverflowAction', 'Saturate');
% 200 in signed Q7.0 is out of range — Wrap gives 200-256 = -56.
disp(fi(200, T, F_wrap));
% Saturate clamps to 127.
disp(fi(200, T, F_sat));
