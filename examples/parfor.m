% parfor — parallel for-loop. Iterations run on a thread pool; scalar
% accumulators (x = x + ...) are compiled as mutex-guarded atomic
% reductions, so the sum is deterministic across thread counts.
%
% === Inline body, no helper function ===
x = 0;
parfor i = 1:10
    x = x + i;       % reduction: 1 + 2 + ... + 10 = 55
end
disp('sum(1..10) via parfor =');
disp(x);

% Two reductions in the same parfor body — each accumulator gets its
% own atomic update site.
a = 0;
b = 0;
parfor i = 1:5
    a = a + i;       % 15
    b = b + 2 * i;   % 30
end
disp('a (sum 1..5)        =');
disp(a);
disp('b (sum of 2*(1..5)) =');
disp(b);

% Non-unit step, still parallel.
s = 0;
parfor k = 2:2:10
    s = s + k;       % 2 + 4 + 6 + 8 + 10 = 30
end
disp('sum of evens in 2..10 =');
disp(s);

% === Body calls out to user functions ===
% The loop body invokes sq(i) (below), so each thread makes a
% standalone call per iteration — exercises the outlined-body +
% function-call path together.
total = 0;
parfor i = 1:5
    total = total + sq(i);    % 1 + 4 + 9 + 16 + 25 = 55
end
disp('sum of squares 1..5 =');
disp(total);

% parfor with a side-effecting helper (I/O per iteration). Order of
% lines across threads is not fixed, but each iteration fires exactly
% once.
disp('per-iteration messages (order may vary across threads):');
parfor i = 1:10
    greet(i);
end

function y = sq(x)
    y = x * x;
end

function greet(i)
    fprintf('  hello from iteration %g\n', i);
end
