% DSP System Toolbox Tier-2 — filter design (function-form).
%
% firpm (Parks-McClellan equiripple) + firls (least-squares) FIR design,
% and the iirnotch / iirpeak second-order designers.  No System Object
% needed — these ship independently of the Tier-1 SO model.
N = 20;
edges = [0 0.3 0.4 1];
amp   = [1 1 0 0];                  % lowpass: pass [0,0.3], stop [0.4,1]

bp = firpm(N, edges, amp);
bl = firls(N, edges, amp);
k   = 0:N;
alt = cos(pi * k);                 % (-1)^k -> Nyquist gain via sum(b.*alt)

fprintf('firpm dc=%.3f nyq=%.3f taps=%d\n', sum(bp), sum(bp .* alt), numel(bp));
fprintf('firls dc=%.3f nyq=%.3f taps=%d\n', sum(bl), sum(bl .* alt), numel(bl));
% linear-phase symmetry: matching outer taps mirror about the centre
fprintf('firpm sym %.4f=%.4f  %.4f=%.4f\n', bp(1), bp(N+1), bp(2), bp(N));

% iirnotch: filter rejects a tone AT the notch frequency, keeps a passband
% tone almost untouched.
[bn, an] = iirnotch(0.5, 0.1);
n = 0:199;
yn = filter(bn, an, sin(0.5 * pi * n));
yp = filter(bn, an, sin(0.15 * pi * n));
fprintf('notch reject=%.4f passband=%.3f\n', ...
        max(abs(yn(150:200))), max(abs(yp(150:200))));
