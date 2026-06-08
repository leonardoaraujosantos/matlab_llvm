% channel_split.m — per-channel RGB work with 3-D indexing + cat(3,...).
% ----------------------------------------------------------------------
% Build a truecolor image from three planes with cat(3,...), pull each
% channel out with rgb(:,:,k), boost one channel, and recombine — the
% canonical "split / process / merge" colour workflow.  Element access
% A(i,j,k) and whole-plane slices A(:,:,k) are both supported.
H = 32; W = 48;

% ----- assemble an RGB image from three 2-D planes --------------------
R = zeros(H, W); G = zeros(H, W); B = zeros(H, W);
for i = 1:H
    for j = 1:W
        R(i, j) = 40 + 4 * i;       % vertical red ramp
        G(i, j) = 30 + 3 * j;       % horizontal green ramp
        B(i, j) = 120;              % flat blue
    end
end
rgb = cat(3, R, G, B);
fprintf('image is %.0fx%.0fx%.0f\n', size(rgb, 1), size(rgb, 2), size(rgb, 3));
fprintf('pixel (1,1) = [%.0f %.0f %.0f]\n', rgb(1,1,1), rgb(1,1,2), rgb(1,1,3));

% ----- split channels with whole-plane slices -------------------------
red   = rgb(:, :, 1);
green = rgb(:, :, 2);
blue  = rgb(:, :, 3);
fprintf('channel means R/G/B = %.1f / %.1f / %.1f\n', mean2(red), mean2(green), mean2(blue));

% ----- boost the red channel and merge back ---------------------------
red2 = imadd(red, 60);              % brighten red (saturating)
out  = cat(3, red2, green, blue);
fprintf('after boost, red mean %.1f -> %.1f\n', mean2(red), mean2(out(:, :, 1)));

% ----- luminance of original vs boosted -------------------------------
fprintf('luminance mean %.1f -> %.1f\n', mean2(rgb2gray(rgb)), mean2(rgb2gray(out)));

% ----- write the colour images (input + red-boosted) ------------------
imwrite(rgb, '/tmp/img_channels_input.png');
imwrite(out, '/tmp/img_channels_boosted.png');
fprintf('wrote /tmp/img_channels_{input,boosted}.png\n');
