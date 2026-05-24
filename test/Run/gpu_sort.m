% T5 gate — gpucoder.sort.  v1 reference is the host qsort fallback;
% backends will swap in CUB radix (CUDA) / bitonic (Metal + OpenCL).
X = [4 1 3 5 2];
Y = gpucoder.sort(X);
disp(Y);
