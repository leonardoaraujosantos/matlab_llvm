% Tier-3 (Signal Processing Toolbox roadmap §4.1): real multirate —
% resample / decimate / interp / upfirdn. Replaces the toy upsample /
% downsample stubs (which lack anti-aliasing) with proper FIR-based
% versions.

% Output-length identities — the most stable thing to test across
% lanes (filter values are transient-dominated for short inputs).
x = 1:32;

% decimate(x, r): output length ceil(N/r).
y2 = decimate(x, 2);   disp(size(y2, 2));   % 16
y4 = decimate(x, 4);   disp(size(y4, 2));   % 8

% interp(x, r): output length N*r.
i2 = interp(x, 2);     disp(size(i2, 2));   % 64
i3 = interp(x, 3);     disp(size(i3, 2));   % 96

% resample(x, p, q): output length ceil(N*p/q).
r32 = resample(x, 3, 2);   disp(size(r32, 2));   % 48
r23 = resample(x, 2, 3);   disp(size(r23, 2));   % ceil(64/3) = 22

% upfirdn p = q = 1 with a length-3 boxcar smoothing FIR.
h = [1 1 1];
y_pass = upfirdn(x, h, 1, 1);
disp(size(y_pass, 2));     % 32 + 3 - 1 = 34
disp(y_pass(1));           % 1
disp(y_pass(2));           % 1 + 2 = 3
disp(y_pass(3));           % 1 + 2 + 3 = 6

% r = 1 passthrough.
yp = decimate(x, 1);
disp(yp(1));    % 1
disp(yp(32));   % 32
