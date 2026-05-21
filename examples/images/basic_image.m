% basic_image.m — Image Processing Toolbox Tier-1.
% ----------------------------------------------------------------------
% The "Basic Image Import, Processing, and Export" workflow: build an
% image, summarise it, adjust its contrast, write it to disk and read it
% back, confirming the round-trip.  Images are double matrices in [0,255]
% (uint8-class) — no external image-library dependency.
%
% A synthetic low-contrast gradient stands in for a loaded photo so the
% example is fully self-contained (imread/imwrite are exercised on the
% PGM round-trip below).
H = 64; W = 64;
img = zeros(H, W);
for i = 1:H
    for j = 1:W
        img(i, j) = 80 + 0.6 * (i + j);     % a dim, low-contrast ramp
    end
end

fprintf('original  mean = %.1f, std = %.1f\n', mean2(img), std2(img));
sl = stretchlim(img);
fprintf('stretchlim     = [%.3f %.3f]\n', sl(1), sl(2));

% ----- contrast enhancement -------------------------------------------
adj = imadjust(img);                        % auto-stretch to full range
fprintf('adjusted  mean = %.1f, std = %.1f\n', mean2(adj), std2(adj));
fprintf('adjusted  range = [%.0f %.0f]\n', min(min(adj)), max(max(adj)));

% ----- write / read round-trip ----------------------------------------
imwrite(adj, '/tmp/basic_image.pgm');
back = imread('/tmp/basic_image.pgm');
fprintf('round-trip max difference = %.0f\n', max(max(imabsdiff(back, adj))));
