% Issue #20 Phase 1 — outer-scope constant captured into a parfor body
% must lower cleanly.  Pre-fix, this errored with:
%   parfor: body captures value of unsupported defining op 'matlab.alloc'
% because SlotPromotion bails on any cross-block use and the outliner
% rejects matlab.alloc on its cloneable-external allowlist.
%
% The ForwardParforCaptures pass clones the constant inside the parfor
% body before OutlineParfor's capture analysis runs, so the outliner
% only sees a cloneable arith.constant on the operand chain.

W = 800;
total = 0;
parfor j = 1:10
    total = total + j * W;
end
% sum_{j=1..10} j * 800 = 800 * 55 = 44000
fprintf('total = %.0f\n', total);

% Two-capture variant that the outliner's reduction detector also
% recognises (j * W * H stays a single matlab.add into the reduction
% store).
H = 600;
two = 0;
parfor j = 1:10
    two = two + j * W * H;
end
% 800 * 600 * 55 = 26 400 000
fprintf('two   = %.0f\n', two);

% Helper-call form: the Mandelbrot-style row dispatcher from issue #20.
% Helper takes the captured scalars by argument, so the parfor body
% only loads `j` and the captures — the same pattern PCT users write.
max_iter = 512;
m_total = 0;
parfor j = 1:600
    m_total = m_total + row_iters(j, W, max_iter);
end
% sum_{j=1..600} (j*800 + 512) = 800*sum(1..600) + 512*600 = 144 547 200
fprintf('mandel = %.0f\n', m_total);

function n = row_iters(j, w, m)
    n = j * w + m;
end
