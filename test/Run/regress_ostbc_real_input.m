% Regression: ostbcCombine on a REAL-matrix input must not read past the buffer.
% An upstream complex op (x*complex(h)) can degrade to a real matrix; reading it
% through the complex layout was a heap-buffer-overflow -> intermittent SIGSEGV
% under -dap (#214). The combiner now accepts a real input (im=0). We check the
% output element count matches the input (N) — garbage N without the fix.
y = [1 2 3 4];
r = ostbcCombine(y, 0.6, 0.2, 0.5, -0.4);
fprintf('n4=%.0f\n', numel(r));
y2 = [1 2 3 4 5 6 7 8];
r2 = ostbcCombine(y2, 1.0, 0.0, 0.0, 0.0);
fprintf('n8=%.0f\n', numel(r2));
