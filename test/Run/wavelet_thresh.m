% thresholding + threshold selection + noise estimation
v = [-3 -1 0.5 2 4];
s = wthresh(v, 's', 1.5);
h = wthresh(v, 'h', 1.5);
fprintf('soft -3->%.2f  4->%.2f\n', s(1), s(5));
fprintf('hard -1->%.2f  4->%.2f\n', h(2), h(5));
thr = thselect(1:64, 'sqtwolog');
fprintf('universal(64): %.3f\n', thr);
mm = thselect(1:64, 'minimaxi');
fprintf('minimax(64): %.3f\n', mm);
