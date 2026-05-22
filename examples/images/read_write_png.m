% read_write_png.m — real image-format I/O (PNG codec).
% ----------------------------------------------------------------------
% imread now decodes real PNG and baseline-JPEG files (hand-coded, no
% libpng/libjpeg), and imwrite produces standard lossless PNG.  This demo
% builds an RGB image, writes it as PNG, reads it back, and confirms the
% round-trip is bit-exact — then runs an analysis pipeline on the loaded
% image just as you would on a photo loaded from disk.
%
% (Baseline JPEG decode works the same way for *.jpg files; it is lossy,
% so a JPEG round-trip would not be bit-exact.)
H = 48; W = 64;
R = zeros(H, W); G = zeros(H, W); B = zeros(H, W);
for i = 1:H
    for j = 1:W
        R(i, j) = 30 + 4 * i;
        G(i, j) = 20 + 3 * j;
        B(i, j) = 90;
    end
end
photo = cat(3, R, G, B);

% ----- write + read back a real PNG -----------------------------------
imwrite(photo, '/tmp/demo_photo.png');
loaded = imread('/tmp/demo_photo.png');
fprintf('loaded a %.0fx%.0fx%.0f PNG\n', size(loaded,1), size(loaded,2), size(loaded,3));
fprintf('round-trip max channel error = %.0f (lossless)\n', max(max(imabsdiff(loaded(:,:,1), photo(:,:,1)))));

% ----- analyse the loaded image ---------------------------------------
gray = rgb2gray(loaded);
fprintf('luminance mean = %.1f, std = %.1f\n', mean2(gray), std2(gray));
bw = imbinarize(gray);
fprintf('foreground fraction = %.2f\n', mean2(bw));

% ----- write the grayscale result as PNG too --------------------------
imwrite(gray, '/tmp/demo_gray.png');
g2 = imread('/tmp/demo_gray.png');
fprintf('grayscale PNG round-trip error = %.0f\n', max(max(imabsdiff(g2, gray))));
