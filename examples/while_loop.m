% while-loop variants. MATLAB's while is conditional-at-the-top with a
% boolean test; it maps 1:1 onto C/C++'s `while (cond) { ... }`.

% Simple accumulator: sum of 1..10 using an explicit counter.
i = 1;
total = 0;
while i <= 10
    total = total + i;
    i = i + 1;
end
disp('sum(1..10) =');
disp(total);

% Break out early once a running product exceeds a threshold. Exercises
% `break` inside a while — emitter falls back to the `while (1) { ... }`
% intermediate form to honour the extra break-slot predicate.
n = 1;
p = 1;
while 1 == 1
    p = p * n;
    if p > 1000
        break;
    end
    n = n + 1;
end
disp('first n where n! > 1000:');
disp(n);
disp('that factorial:');
disp(p);

% Classic Newton-Raphson for sqrt(2), driven by a convergence check
% rather than a fixed iteration count — the natural shape for while.
x = 1.0;
while abs(x * x - 2) > 1e-12
    x = (x + 2 / x) / 2;
end
disp('Newton-Raphson sqrt(2):');
disp(x);
