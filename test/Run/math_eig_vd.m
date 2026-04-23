A = [4 1; 1 3];
[V, D] = eig(A);
disp(V * D * V');
