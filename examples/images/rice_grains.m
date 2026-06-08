% rice_grains.m — Image Processing Toolbox Tier-5 HEADLINE.
% ----------------------------------------------------------------------
% The Getting-Started "Correct Nonuniform Illumination and Analyze
% Foreground Objects" workflow (the MATLAB rice.png demo):
%   1. estimate the uneven background with a morphological opening
%   2. subtract it to flatten the illumination
%   3. threshold (Otsu) to a binary mask
%   4. label the connected components and measure each one.
%
% A synthetic rice-grain field (bright blobs over a brightness ramp)
% stands in for rice.png so the example is fully self-contained.
H = 160; W = 200;

% ----- build the test image: bright grains over uneven illumination ----
I = zeros(H, W);
for i = 1:H
    for j = 1:W
        I(i, j) = 50 + 0.35 * (i + j);          % diagonal brightness ramp
    end
end
ng = 0;
for gi = 1:5
    for gj = 1:8
        cy = 15 + gi * 26;
        cx = 12 + gj * 23;
        for di = -4:4
            for dj = -2:2
                yy = cy + di; xx = cx + dj;
                if yy >= 1 && yy <= H && xx >= 1 && xx <= W
                    I(yy, xx) = 235;            % a grain
                end
            end
        end
        ng = ng + 1;
    end
end
fprintf('placed %.0f grains over a brightness ramp\n', ng);

% ----- 1-2. flatten the illumination by background subtraction ---------
bg   = imopen(I, strel('disk', 12));            % opening removes the grains
flat = imsubtract(I, bg);
fprintf('background range before flatten = [%.0f %.0f]\n', min(min(I)), max(max(I)));
fprintf('after subtraction, background ~ %.1f\n', mean2(imopen(flat, strel('disk', 12))));

% ----- 3. threshold to a binary mask ----------------------------------
level = graythresh(flat);
BW    = imbinarize(flat, level);
BW    = bwareaopen(BW, 8);                       % drop specks
fprintf('Otsu level = %.3f\n', level);

% ----- 4. label + measure ---------------------------------------------
L = bwlabel(BW);
n = max(max(L));
areas = regionprops(L, 'Area');
fprintf('grains detected = %.0f\n', n);
fprintf('mean grain area = %.1f px\n', mean(areas));
fprintf('min / max area  = %.0f / %.0f px\n', min(areas), max(areas));

% ----- write the pipeline result images -------------------------------
imwrite(I,            '/tmp/img_rice_input.png');      % uneven illumination
imwrite(flat,         '/tmp/img_rice_flattened.png');  % after background subtraction
imwrite(BW .* 255,    '/tmp/img_rice_mask.png');       % Otsu binary mask
imwrite(label2rgb(L), '/tmp/img_rice_labels.png');     % coloured components
fprintf('wrote /tmp/img_rice_{input,flattened,mask,labels}.png\n');
