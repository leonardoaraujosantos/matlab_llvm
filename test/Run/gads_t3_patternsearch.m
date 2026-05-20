% Global Optimization Tier-3 — patternsearch (direct search).
% On a discontinuous staircase bowl the FD gradient is ~0 on each flat
% step, so a gradient solver stalls; patternsearch compares function
% values only and steps down to the global minimum.  Also a smooth
% sanity case.  (x*x avoids the scalar-matpow-in-anon gap.)
f = @(x) round(((x(1)-2)*(x(1)-2) + (x(2)+3)*(x(2)+3)) * 2) / 2;
xu = fminunc(f, [7; 7]);
fprintf('fminunc       f = %.4f\n', f(xu));   % 125.0000 (stalled)
xp = patternsearch(f, [7; 7], [-10; -10], [10; 10]);
fprintf('patternsearch f = %.4f\n', f(xp));   %   0.0000 (global)
fprintf('x1 = %.2f\n', xp(1));                %   2.00
fprintf('x2 = %.2f\n', xp(2));                %  -3.00

% Nonsmooth V-valley, min at the kink (1,-2), f=0.5.
g = @(x) 3*abs(x(1)-1) + 3*abs(x(2)+2) + 0.5;
y = patternsearch(g, [5; 5], [-10; -10], [10; 10]);
fprintf('vvalley f = %.4f\n', g(y));          % 0.5000
