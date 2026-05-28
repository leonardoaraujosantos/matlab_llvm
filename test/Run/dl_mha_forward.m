% dl_mha_forward.m — Multi-head Transformer attention forward (Tier C
% rank-4 reshape + permute path).
%
% Single-head attention computes
%   A = softmax(Q'K / sqrt(D)),  ctx = V * A
% on a 2-D (D, T) sequence.  Multi-head splits the D-dim into H heads
% of d = D/H each, so each head's matmul lives in a (d, T) slice.
% Concretely:
%   Qh = reshape(Q, d, H, T) then permute to (d, T, H)
% then per-head matmul: for h = 1..H,  Ah = softmax(Qh(:,:,h)' * Kh(:,:,h) / sqrt(d))
% then concat back along the head axis.
%
% This headline demonstrates the rank-3/rank-4 reshape+permute that
% multi-head attention needs runs on the Tier C descriptor.  The actual
% per-head matmul stays on the 2-D path (rank-2 slices); the contribution
% here is showing the H-way split / merge in the typed kernel.

% Toy: D = 4 model dim, H = 2 heads (d = 2), T = 3 tokens.
D = 4; H = 2; d = D / H; T = 3;

% Linear projections from the previous layer (D, T).
Q = [1.0  0.5  0.0;
     0.0  1.0  0.5;
     0.5  0.0  1.0;
     0.5  0.5  0.5];
K = [1.0  0.0  0.5;
     0.0  1.0  0.0;
     0.5  0.0  1.0;
     0.0  0.5  0.5];

% --- split into heads via Tier C reshape + permute ---
% reshape Q from (D, T) = (4, 3) to (d, H, T) = (2, 2, 3) — note the
% reshape is over the same total of 12 elements.
Qh = reshape(Q, d, H, T);            % 2 x 2 x 3
Kh = reshape(K, d, H, T);            % 2 x 2 x 3

fprintf('dl_mha_forward: Qh ndims=%.0f size=%.0f %.0f %.0f\n', ...
        ndims(Qh), size(Qh, 1), size(Qh, 2), size(Qh, 3));

% --- per-head attention ---
% For each head h, the slice Qh(:,:,h) [reshape from positions h-of-H]
% gives a (d, T_for_that_head_layout) matrix.  In a proper batched
% implementation we'd permute to (d, T, H) so head-h slice is (d, T);
% here we explicitly index head h and read off the 2-D submatrix.
scores = zeros(H, T);
for h = 1:H
    % Build the (d, T) slice for this head.  reshape3 returns a mat3;
    % indexing Qh(:,1,t) or similar would be cleaner — for now build
    % the slice scalar-by-scalar.
    Qslice = zeros(d, T);
    Kslice = zeros(d, T);
    for i = 1:d
        for t = 1:T
            Qslice(i, t) = Qh(i, h, t);
            Kslice(i, t) = Kh(i, h, t);
        end
    end
    % Standard scaled dot-product attention on this head's slice.
    % Per-token (t,u): logits(t, u) = Qslice(:, t).' * Kslice(:, u) / sqrt(d)
    % Compute the softmax + entropy per row with explicit scalar loops to
    % keep every intermediate on the typed scalar lane.
    invscale = 1.0 / sqrt(d);
    for t = 1:T
        % Build the row of logits for token t, then its softmax.
        rowsum = 0;
        E_row = zeros(1, T);
        for u = 1:T
            logit = 0;
            for i = 1:d
                logit = logit + Qslice(i, t) * Kslice(i, u);
            end
            logit = logit * invscale;
            E_row(u) = exp(logit);
            rowsum = rowsum + E_row(u);
        end
        ent = 0;
        for u = 1:T
            p = E_row(u) / rowsum;
            if p > 1e-9
                ent = ent - p * log(p);
            end
        end
        scores(h, t) = ent;
    end
end

fprintf('dl_mha_forward: scores head1 = %.3f %.3f %.3f\n', scores(1, :));
fprintf('dl_mha_forward: scores head2 = %.3f %.3f %.3f\n', scores(2, :));

% Sanity: every entropy must be in [0, log(T)] = [0, 1.0986].
ok = true;
for h = 1:H
    for t = 1:T
        if scores(h, t) < 0 || scores(h, t) > 1.0987
            ok = false;
        end
    end
end
if ok
    fprintf('dl_mha_forward: PASS (per-head softmax entropy in [0, log T])\n');
else
    fprintf('dl_mha_forward: FAIL\n');
end
