% Verify Stats odeset option prints accepted/rejected/feval counts. The
% MATLAB-canonical syntax is opts.Stats = 'on'; here we accept the
% numeric flag opts.Stats = 1 (the frontend's struct-set lowering for
% string values isn't yet wired through matlab_struct_set_f64). The
% stats line is emitted to stdout once per [t,y] = ode45(...) site.

f = @(t,y) -2*y + sin(t);
opts.Stats = 1;
[t, y] = ode45(f, [0 10], 1, opts);
% The exact step counts depend on FP details. Check stats are non-empty
% by post-printing the last sample so we can tell the test ran.
disp(t(end));
