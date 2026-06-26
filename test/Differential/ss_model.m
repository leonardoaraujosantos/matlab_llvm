% CST model object (ss) — handle/object classdef via the runtime.
A = [0 1; -2 -3]; B = [0; 1]; C = [1 0]; D = 0;
sys = ss(A, B, C, D);
disp('ss ok');
