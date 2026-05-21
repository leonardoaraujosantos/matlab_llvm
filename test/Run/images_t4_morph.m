% Image Processing Tier-4 — binarization + morphology + edges.
S = zeros(12, 12); S(4:9, 4:9) = 1;          % a 6x6 square
fprintf('graythresh %.3f\n', graythresh([0 0 0 255 255 255]));
se = strel('square', 3);
E = imerode(S, se);  D = imdilate(S, se);
fprintf('erode  %.0f  dilate %.0f  src %.0f\n', sum(sum(E)), sum(sum(D)), sum(sum(S)));
O = imopen(S, se);   C = imclose(S, se);
fprintf('open   %.0f  close  %.0f\n', sum(sum(O)), sum(sum(C)));
ring = S; ring(6:7, 6:7) = 0;                 % square with a hole
F = imfill(ring);
fprintf('hole filled back to %.0f\n', sum(sum(F)));
G = zeros(10,10); G(:,6:10) = 200;
ed = edge(G, 'sobel');
fprintf('edge sum %.0f  canny %.0f\n', sum(sum(ed)), sum(sum(edge(G,'canny'))));
op = bwareaopen(S, 50);
fprintf('areaopen50 %.0f\n', sum(sum(op)));
