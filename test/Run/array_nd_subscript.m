% Fewer-subscript indexing of an N-D array (previously segfaulted).
% A(i,j) on an M×N×P array collapses the trailing dims into the last
% subscript (MATLAB's rule): j spans N*P with dim 2 fastest, returning the
% logical element A(i,n,k).  A(lin) is a linear index over the flat
% slice-major buffer (the project's documented order).
A = zeros(2, 2, 3);
A(1,1,1) = 11;
A(2,2,3) = 99;
A(1,2,2) = 55;

% 2-subscript reads: A(2,2) -> A(2,2,1) = 0 (only A(2,2,3) was set)
fprintf('ij %.0f %.0f\n', A(1,1), A(2,2));

% trailing-dim collapse: A(1,4) -> n=(4-1)%2=1, k=(4-1)/2=1 -> A(1,2,2)=55
fprintf('collapse %.0f\n', A(1,4));

% linear index over the flat buffer: A(1)=first, A(12)=last element
fprintf('lin %.0f %.0f\n', A(1), A(12));
