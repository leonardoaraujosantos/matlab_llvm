% for-loop fixture: the for-stmt anchors a hook on the `for` keyword;
% the loop body's statements emit their own hooks once each (the IR
% emits a single static hook inside the body region, executed per
% iteration at runtime).
n = 3;

for i = 1:n
    % loop-internal comment
    x = i * 2;
end

disp(x);
