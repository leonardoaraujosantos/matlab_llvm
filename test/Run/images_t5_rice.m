% Image Processing Tier-5 — rice-grain pipeline (illumination + count).
H = 100; W = 120; I = zeros(H, W);
for i = 1:H
    for j = 1:W
        I(i, j) = 40 + 0.4 * (i + j);
    end
end
for gi = 1:3
    for gj = 1:5
        cy = 15 + gi * 24; cx = 12 + gj * 22;
        for di = -3:3
            for dj = -2:2
                I(cy + di, cx + dj) = 230;
            end
        end
    end
end
bg = imopen(I, strel('disk', 10));
flat = imsubtract(I, bg);
BW = imbinarize(flat, graythresh(flat));
BW = bwareaopen(BW, 6);
L = bwlabel(BW);
fprintf('grains %.0f\n', max(max(L)));
a = regionprops(L, 'Area');
fprintf('meanarea %.1f\n', mean(a));
