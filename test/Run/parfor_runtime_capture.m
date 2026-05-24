% Issue #20 Phase 2 — outer-scope scalar that ForwardParforCaptures
% can't const-forward (multi-store / non-literal value) must still
% reach the parfor body via the state[] tail extension in LowerParfor.
%
% This complements parfor_scalar_capture.m (Phase 1: single-literal
% const forwarding) by exercising the runtime-computed code path:
% LowerParfor emits one matlab.load(%slot) at the dispatch site, stores
% the loaded value into state[NRed+k], and the outlined function reads
% it back at the captured type.

% W is f64-typed at OutlineParfor time (all stores are f64), but
% ForwardParforCaptures bails because there are multiple stores —
% Phase 2's state[] path picks it up.
W = 800;
for k = 1:3
    W = W + 100;
end
total = 0;
parfor j = 1:10
    total = total + j * W;
end
% After the for loop W = 1100; sum_{j=1..10} j*1100 = 60500
fprintf('total = %.0f\n', total);

% Two captures in the same parfor: W (mutated above) and a
% second slot whose value is the product of two captures.
% Reduction chain stays a single add: load(acc) + (j*W*Z).
Z = 2;
for k = 1:3
    Z = Z + 1;   % Z = 5
end
acc = 0;
parfor j = 1:10
    acc = acc + j * W * Z;
end
% W=1100, Z=5; sum_{j=1..10} j*1100*5 = 5500 * 55 = 302500
fprintf('acc   = %.0f\n', acc);
