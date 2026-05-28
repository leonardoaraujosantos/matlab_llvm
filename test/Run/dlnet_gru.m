% Deep Learning T4 gating test — functional GRU with BPTT through the
% reset/update gate path (canonical pitfall: the candidate's recurrent
% contribution sees `r .* h_prev`, not h_prev directly — the test verifies
% that gradient flows through that elementwise gate).

D = 2; H = 3; T = 4;

Xd = [ 1.0 -0.5  0.2  0.8;
       0.3  0.7 -0.1  0.4 ];
Td = [ 0.1  0.2  0.3  0.4;
      -0.1  0.0  0.1  0.2;
       0.5 -0.2  0.3  0.0 ];

Wx_init = 0.1 * ones(3*H, D);
Wr_init = 0.1 * ones(3*H, H);
b_init  = zeros(3*H, 1);

X  = dlarray(Xd);
h0 = dlarray(zeros(H, 1));
Wx = dlarray(Wx_init);
Wr = dlarray(Wr_init);
b  = dlarray(b_init);

Hseq = gru(X, h0, Wx, Wr, b);
Tt   = dlarray(Td);
loss = mse(Hseq, Tt);
L0v  = extractdata(loss); L0 = L0v(1);

gWx = dlgradient(loss, Wx);
gWr = dlgradient(loss, Wr);
gb  = dlgradient(loss, b);

lr = 0.2;
Wx2 = dlarray(Wx_init - lr * gWx);
Wr2 = dlarray(Wr_init - lr * gWr);
b2  = dlarray(b_init  - lr * gb);
Hseq2 = gru(X, h0, Wx2, Wr2, b2);
loss2 = mse(Hseq2, Tt);
L1v   = extractdata(loss2); L1 = L1v(1);

gWx_m = sum(sum(gWx .* gWx));
gWr_m = sum(sum(gWr .* gWr));
gb_m  = sum(gb .* gb);

Hseq_d = extractdata(Hseq);
shape_ok = 0;
if size(Hseq_d, 1) == H
    if size(Hseq_d, 2) == T
        shape_ok = 1;
    end
end
grad_nz = 0;
if gWx_m > 0
    if gWr_m > 0
        if gb_m > 0
            grad_nz = 1;
        end
    end
end
loss_drop = 0;
if L1 < L0
    loss_drop = 1;
end

fprintf('gru shape ok = %.0f\n', shape_ok);
fprintf('gradient nonzero = %.0f\n', grad_nz);
fprintf('loss drops after BPTT step = %.0f\n', loss_drop);
