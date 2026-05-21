% Image Processing Tier-5 — segmentation + region analysis.
BW = zeros(20, 20);
BW(2:5, 2:5) = 1; BW(2:6, 12:15) = 1; BW(12:15, 7:10) = 1;
L = bwlabel(BW);
fprintf('numobj %.0f\n', max(max(L)));
ar = regionprops(L, 'Area');
fprintf('areas %.0f %.0f %.0f\n', ar(1), ar(2), ar(3));
ct = regionprops(L, 'Centroid');
fprintf('cent1 %.1f %.1f\n', ct(1,1), ct(1,2));
bb = regionprops(L, 'BoundingBox');
fprintf('bbox1 %.1f %.1f %.0f %.0f\n', bb(1,1), bb(1,2), bb(1,3), bb(1,4));
ed = regionprops(L, 'EquivDiameter');
fprintf('eqd1 %.2f\n', ed(1));
fprintf('euler %.0f\n', bweuler(BW));
rgb = label2rgb(L);
fprintf('label2rgb %.0fx%.0fx%.0f\n', size(rgb,1), size(rgb,2), size(rgb,3));
op = bwareaopen(BW, 18);
fprintf('areaopen18 sum %.0f\n', sum(sum(op)));
