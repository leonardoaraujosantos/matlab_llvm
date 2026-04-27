% Fixture for the data-breakpoint (watchpoint) scenario. The DAP
% client sets a write-watch on `target` before configurationDone;
% the runtime trips on line 4 (the second assignment) and the
% scenario verifies the stopped event reports reason="data
% breakpoint" plus the watch's id in hitBreakpointIds.
target = 1;
target = 2;
disp(target);
