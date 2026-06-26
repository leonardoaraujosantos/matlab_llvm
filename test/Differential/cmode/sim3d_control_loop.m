% Handle classdef + loop-carried matrix accumulation in one program.
w = sim3d.World();
cart = sim3d.Actor('cart', 'box');
w.add(cart);
w.open();
X = [0; 0; 0.2; 0];
K = [-4 -5.77 -50.15 -11.71];
for k = 1:40
    u = -K * X;
    X = X + 0.02 * [X(2); u; X(4); -u];
    cart.Translation = [X(1) 0 0.1];
    w.run(0.02);
    if mod(k,10)==0
        fprintf('k=%d x=%.5f theta=%.5f\n', k, X(1), X(3));
    end
end
w.close();
disp('done');
