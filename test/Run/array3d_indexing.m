% 3-D array indexing: element + slice read/store, and cat(1/2/3,...).
A = zeros(4, 5, 3);
A(:,:,1) = 10; A(:,:,2) = 20; A(:,:,3) = 30;   % slice store (scalar broadcast)
A(2,3,2) = 99;                                  % element store
fprintf('elem %.0f\n', A(2,3,2));
P = A(:,:,3);                                   % slice read -> 2-D plane
fprintf('plane %.0fx%.0f v %.0f\n', size(P,1), size(P,2), P(1,1));
M = ones(4,5) * 7;
A(:,:,1) = M;                                   % slice store from a matrix
fprintf('matstore %.0f\n', A(1,1,1));
% cat(3,...) build truecolor, then index channels
R = ones(3,3)*100; G = ones(3,3)*150; Bc = ones(3,3)*200;
rgb = cat(3, R, G, Bc);
fprintf('cat3 %.0fx%.0fx%.0f\n', size(rgb,1), size(rgb,2), size(rgb,3));
fprintf('chans %.0f %.0f %.0f\n', rgb(1,1,1), rgb(1,1,2), rgb(1,1,3));
fprintf('gray %.1f\n', max(max(rgb2gray(rgb))));
% cat dim 1/2 + 2-arg dim-3
fprintf('cat12 %.0f %.0f\n', size(cat(1,R,G),1), size(cat(2,R,G),2));
fprintf('cat3-2 %.0f\n', size(cat(3,R,G),3));
