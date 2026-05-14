% regress_anon_call_after_pass.m — regression test for calling an
% anonymous-function handle directly after it has also been passed as
% an argument to a builtin.  Before the fix, `ros(r)` left an
% unlowered matlab.call_indirect: the anon outliner only rewrote
% call_indirect uses that were *direct* users of the addressof, and
% the post-pass call-indirect resolver only traced through llvm.load
% slots, not the matlab.load slots that survive at that stage.  The
% post-pass now traces matlab.load too and bridges tensor/ptr operand
% mismatches.

% --- 1. anon passed to fminsearch, then called directly ----------
ros = @(x) 100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + ...
           (1 - x(1))*(1 - x(1));
r = fminsearch(ros, [-1.2; 1]);
fval = ros(r);                       % direct call of the passed anon
if fval < 1e-6; disp(1); else; disp(0); end

% --- 2. same pattern with fminunc --------------------------------
bowl = @(x) (x(1) - 2)*(x(1) - 2) + (x(2) + 3)*(x(2) + 3);
q = fminunc(bowl, [0; 0]);
qval = bowl(q);                      % direct call again
if qval < 1e-8; disp(1); else; disp(0); end

% --- 3. scalar anon passed to fzero, then called directly --------
f = @(x) cos(x) - x;
root = fzero(f, 0.5);
resid = f(root);                     % direct scalar call
if abs(resid) < 1e-9; disp(1); else; disp(0); end

% --- 4. the directly-called value is usable downstream -----------
chk = ros([1; 1]);                   % Rosenbrock minimum is exactly 0
if chk < 1e-12; disp(1); else; disp(0); end
