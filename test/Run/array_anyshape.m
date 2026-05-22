% Tier A (any_shape): arbitrary-depth 3-D made first-class.
% N-ary cat(3,…) of any depth + 3-D-ness that survives expressions
% (expression-aware ThreeDBindings, not a syntactic allowlist).
% fprintf caps at fmt+3 values here, so wide prints are split.

% --- arbitrary depth, beyond the old cap of 3 ---
A = zeros(2, 3, 4);
for k = 1:4
  A(:,:,k) = k * 10;          % whole-plane store, any plane
end
fprintf('A %.0fx%.0fx%.0f\n', size(A,1), size(A,2), size(A,3));
fprintf('A n=%.0f d=%.0f\n', numel(A), ndims(A));
fprintf('Aplanes %.0f %.0f %.0f\n', A(1,1,1), A(1,1,2), A(1,1,3));
fprintf('Aplane4 %.0f\n', A(1,1,4));

Z = zeros(2, 2, 8);
Z(2,2,8) = 77;                 % element store on plane 8
fprintf('Z d=%.0f n=%.0f last=%.0f\n', size(Z,3), numel(Z), Z(2,2,8));

% --- N-ary cat(3,…): 4 and 5 planes (was silently capped at depth 3) ---
p1 = ones(2,2)*1; p2 = ones(2,2)*2; p3 = ones(2,2)*3; p4 = ones(2,2)*4;
C4 = cat(3, p1, p2, p3, p4);
fprintf('cat4 d=%.0f\n', size(C4,3));
fprintf('cat4vals %.0f %.0f %.0f\n', C4(1,1,1), C4(1,1,2), C4(1,1,3));
fprintf('cat4last %.0f\n', C4(1,1,4));
C5 = cat(3, p1, p2, p3, p4, ones(2,2)*5);
fprintf('cat5 d=%.0f last=%.0f n=%.0f\n', size(C5,3), C5(1,1,5), numel(C5));

% --- 3-D-ness survives expressions (the documented A2 case) ---
B = ones(5,5,4) * 3;           % scalar mul preserves depth
fprintf('expr d=%.0f v=%.0f\n', size(B,3), B(1,1,4));
S = 10 + ones(3,3,4);          % scalar-on-left preserves depth
fprintf('sm d=%.0f v=%.0f\n', size(S,3), S(2,2,4));
G = B;                         % aliasing keeps 3-D
fprintf('alias d=%.0f v=%.0f\n', size(G,3), G(1,1,4));

% --- 2-D fallback unchanged ---
M = zeros(3, 4);
fprintf('2d %.0fx%.0f\n', size(M,1), size(M,2));
fprintf('2d n=%.0f d=%.0f\n', numel(M), ndims(M));
N2 = ones(3,3) * 5;
fprintf('2dexpr %.0f d=%.0f\n', N2(2,2), ndims(N2));

% --- A3: depth-aware elementwise ops on mat3 ---
Aa = ones(2,2,3) * 4;
Bb = ones(2,2,3) * 10;
SS = Aa + Bb;   fprintf('mmadd d=%.0f v=%.0f\n', size(SS,3), SS(1,1,3));
PP = Aa .* Bb;  fprintf('mmemul v=%.0f\n', PP(1,1,3));
NN = -Aa;       fprintf('neg d=%.0f v=%.0f\n', size(NN,3), NN(1,1,3));
GG = Aa > Bb;   fprintf('cmp d=%.0f v=%.0f\n', size(GG,3), GG(1,1,1));

% --- A3: depth-aware image arithmetic on truecolor (depth 3) ---
r = ones(2,2)*50; g = ones(2,2)*100; bch = ones(2,2)*150;
rgb = cat(3, r, g, bch);
ic = imcomplement(rgb);
fprintf('imcompl d=%.0f c1=%.0f c3=%.0f\n', size(ic,3), ic(1,1,1), ic(1,1,3));
ia = imadd(rgb, rgb);
fprintf('imadd d=%.0f v=%.0f\n', size(ia,3), ia(1,1,3));

% --- A5: trailing-singleton drop (zeros(m,n,1) and cat(3,a) are 2-D) ---
Z1 = zeros(3,4,1);
fprintf('z1 d=%.0f s3=%.0f n=%.0f\n', ndims(Z1), size(Z1,3), numel(Z1));
T1 = Z1';                       % 2-D op interop (would segfault on a mat3)
fprintf('z1t %.0fx%.0f d=%.0f\n', size(T1,1), size(T1,2), ndims(T1));
C1 = cat(3, ones(2,2)*5);       % single plane stays 2-D
fprintf('cat1 d=%.0f s3=%.0f v=%.0f\n', ndims(C1), size(C1,3), C1(2,2));
