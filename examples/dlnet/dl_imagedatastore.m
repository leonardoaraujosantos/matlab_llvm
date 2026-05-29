% dl_imagedatastore — T1.8 gating: imageDatastore + countEachLabel +
% splitEachLabel.
%
% Synthesises a /tmp/imds_test/{red,green,blue}/ class folder layout by
% writing tiny RGB PPMs via imwrite, then walks it with imageDatastore
% and checks per-class counts before/after splitEachLabel.

mkdir('/tmp/imds_test/red');
mkdir('/tmp/imds_test/green');
mkdir('/tmp/imds_test/blue');

% Synthesise 4 tiny 2x2 PPMs per class (RGB, range [0, 255]).
img = zeros(2, 2, 3);
for k = 1:4
    % Red.
    img(:, :, 1) = 200 + k; img(:, :, 2) = 0; img(:, :, 3) = 0;
    fn = sprintf('/tmp/imds_test/red/r%d.ppm', k);
    imwrite(img, fn);
    % Green.
    img(:, :, 1) = 0; img(:, :, 2) = 200 + k; img(:, :, 3) = 0;
    fn = sprintf('/tmp/imds_test/green/g%d.ppm', k);
    imwrite(img, fn);
    % Blue.
    img(:, :, 1) = 0; img(:, :, 2) = 0; img(:, :, 3) = 200 + k;
    fn = sprintf('/tmp/imds_test/blue/b%d.ppm', k);
    imwrite(img, fn);
end

ds = imageDatastore('/tmp/imds_test');
counts = countEachLabel(ds);

fprintf('dl_imagedatastore: count blue  = %.0f\n', counts(1));
fprintf('dl_imagedatastore: count green = %.0f\n', counts(2));
fprintf('dl_imagedatastore: count red   = %.0f\n', counts(3));

% Take the first 50% of each label group for training.
trainN = splitEachLabel(ds, 0.5);
counts_after = countEachLabel(ds);

fprintf('dl_imagedatastore: train total = %.0f\n', trainN);
fprintf('dl_imagedatastore: split blue  = %.0f\n', counts_after(1));
fprintf('dl_imagedatastore: split green = %.0f\n', counts_after(2));
fprintf('dl_imagedatastore: split red   = %.0f\n', counts_after(3));

% Materialize 1x1 matrix returns to f64 via (1) subscript so the && lowers.
c1 = counts(1); c2 = counts(2); c3 = counts(3);
ca1 = counts_after(1); ca2 = counts_after(2); ca3 = counts_after(3);
tN = trainN(1);
ok_total = (c1 == 4) && (c2 == 4) && (c3 == 4);
ok_train = (tN == 6);
ok_split = (ca1 == 2) && (ca2 == 2) && (ca3 == 2);
if ok_total && ok_train && ok_split
    fprintf('dl_imagedatastore: PASS\n');
else
    fprintf('dl_imagedatastore: FAIL\n');
end
