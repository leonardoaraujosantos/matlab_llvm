% T5 gate — gpucoder.reduce with anon binary op.  Reduces a 1x5 row
% to a scalar via @(a,b) a+b, exercising the function-handle ABI through
% the LowerTensorOps dispatch into matlab_gpucoder_reduce.
X = [1 2 3 4 5];
addfn = @(a,b) a + b;
s = gpucoder.reduce(X, addfn);
fprintf('reduce-sum = %g\n', s);
prodfn = @(a,b) a * b;
p = gpucoder.reduce(X, prodfn);
fprintf('reduce-prod = %g\n', p);
