% dl_real_image_pipeline.m — Cross-toolbox CNN inference on actual PNGs.
%
% Closes the image-toolbox tie-up.  Pipeline:
%   1. Synthesize three colour test images on the numeric lane.
%   2. imwrite each as a real PNG file (round-tripping through the
%      shipped PNG encoder).
%   3. imread the files back as mat3 (HxWx3) — exercising the real PNG
%      decoder, NOT in-memory synthesis.
%   4. cat(4, ...) into a rank-4 batch.
%   5. Run pretrained colour-CNN inference (conv2d_full -> batchnorm ->
%      relu -> avgpool2d -> reshape -> FC -> softmax).
%   6. Verify each PNG classifies into its target colour bucket.

% ===== Step 1: synthesize 6x6 colour test images =====
% Build each as a mat3 (HxWx3) with values in [0, 255] for imwrite.
img_r = zeros(6, 6, 3);
img_g = zeros(6, 6, 3);
img_b = zeros(6, 6, 3);
for h = 1:6
    for w = 1:6
        img_r(h, w, 1) = 255;
        img_g(h, w, 2) = 255;
        img_b(h, w, 3) = 255;
    end
end

% ===== Step 2: write to actual PNG files =====
imwrite(img_r, '/tmp/dl_red.png');
imwrite(img_g, '/tmp/dl_green.png');
imwrite(img_b, '/tmp/dl_blue.png');

% ===== Step 3: imread back from disk =====
% imread returns a uint8-range [0, 255] mat3.  Scale to [0, 1] before
% feeding into the CNN so we get bit-equivalent behaviour to the
% pretrained_inference demo's in-memory synthesis.
raw_r = imread('/tmp/dl_red.png');
raw_g = imread('/tmp/dl_green.png');
raw_b = imread('/tmp/dl_blue.png');
fprintf('dl_real_image_pipeline: imread shape = %.0f %.0f %.0f\n', ...
        size(raw_r, 1), size(raw_r, 2), size(raw_r, 3));

% Scale 255 -> 1.  Use im2double which handles the conversion.
nr = raw_r / 255;
ng = raw_g / 255;
nb = raw_b / 255;

% ===== Step 4: cat(4, mat3, mat3, mat3) -> rank-4 batch =====
batch = cat(4, nr, ng, nb);
fprintf('dl_real_image_pipeline: batch size = %.0f %.0f %.0f %.0f\n', ...
        size(batch, 1), size(batch, 2), size(batch, 3), size(batch, 4));

% ===== Step 5: pretrained CNN inference =====
% Re-use the same hand-loaded weights from dl_pretrained_inference.m;
% kept inline here to make the demo self-contained.
Wconv = zeros(3, 3, 3, 4);
Wconv(1,1,1,1)= 0.149944; Wconv(1,2,1,1)= 0.129905; Wconv(1,3,1,1)= 0.145504;
Wconv(2,1,1,1)= 0.156777; Wconv(2,2,1,1)= 0.132703; Wconv(2,3,1,1)= 0.152449;
Wconv(3,1,1,1)= 0.149944; Wconv(3,2,1,1)= 0.129905; Wconv(3,3,1,1)= 0.145504;
Wconv(1,1,2,1)= 0.042581; Wconv(1,2,2,1)= 0.036781; Wconv(1,3,2,1)= 0.037906;
Wconv(2,1,2,1)= 0.052175; Wconv(2,2,2,1)= 0.044490; Wconv(2,3,2,1)= 0.047563;
Wconv(3,1,2,1)= 0.042581; Wconv(3,2,2,1)= 0.036781; Wconv(3,3,2,1)= 0.037906;
Wconv(1,1,3,1)= 0.007221; Wconv(1,2,3,1)= 0.008868; Wconv(1,3,3,1)= 0.007221;
Wconv(2,1,3,1)= 0.008868; Wconv(2,2,3,1)= 0.010873; Wconv(2,3,3,1)= 0.008868;
Wconv(3,1,3,1)= 0.007221; Wconv(3,2,3,1)= 0.008868; Wconv(3,3,3,1)= 0.007221;
Wconv(1,1,1,2)=-0.080145; Wconv(1,2,1,2)=-0.073051; Wconv(1,3,1,2)=-0.051773;
Wconv(2,1,1,2)=-0.092336; Wconv(2,2,1,2)=-0.085260; Wconv(2,3,1,2)=-0.062392;
Wconv(3,1,1,2)=-0.080145; Wconv(3,2,1,2)=-0.073051; Wconv(3,3,1,2)=-0.051773;
Wconv(1,1,2,2)= 0.142077; Wconv(1,2,2,2)= 0.140420; Wconv(1,3,2,2)= 0.103950;
Wconv(2,1,2,2)= 0.150675; Wconv(2,2,2,2)= 0.147827; Wconv(2,3,2,2)= 0.100619;
Wconv(3,1,2,2)= 0.142077; Wconv(3,2,2,2)= 0.140420; Wconv(3,3,2,2)= 0.103950;
Wconv(1,1,3,2)= 0.000103; Wconv(1,2,3,2)= 0.000253; Wconv(1,3,3,2)= 0.000103;
Wconv(2,1,3,2)= 0.000253; Wconv(2,2,3,2)= 0.000452; Wconv(2,3,3,2)= 0.000253;
Wconv(3,1,3,2)= 0.000103; Wconv(3,2,3,2)= 0.000253; Wconv(3,3,3,2)= 0.000103;
Wconv(1,1,1,3)=-0.004402; Wconv(1,2,1,3)=-0.005639; Wconv(1,3,1,3)=-0.004402;
Wconv(2,1,1,3)=-0.005639; Wconv(2,2,1,3)=-0.007176; Wconv(2,3,1,3)=-0.005639;
Wconv(3,1,1,3)=-0.004402; Wconv(3,2,1,3)=-0.005639; Wconv(3,3,1,3)=-0.004402;
Wconv(1,1,2,3)=-0.004402; Wconv(1,2,2,3)=-0.005639; Wconv(1,3,2,3)=-0.004402;
Wconv(2,1,2,3)=-0.005639; Wconv(2,2,2,3)=-0.007176; Wconv(2,3,2,3)=-0.005639;
Wconv(3,1,2,3)=-0.004402; Wconv(3,2,2,3)=-0.005639; Wconv(3,3,2,3)=-0.004402;
Wconv(1,1,3,3)= 0.122623; Wconv(1,2,3,3)= 0.155572; Wconv(1,3,3,3)= 0.162644;
Wconv(2,1,3,3)= 0.122802; Wconv(2,2,3,3)= 0.163088; Wconv(2,3,3,3)= 0.170939;
Wconv(3,1,3,3)= 0.122623; Wconv(3,2,3,3)= 0.155572; Wconv(3,3,3,3)= 0.162644;
Wconv(1,1,1,4)= 0.125431; Wconv(1,2,1,4)= 0.108992; Wconv(1,3,1,4)= 0.120018;
Wconv(2,1,1,4)= 0.138437; Wconv(2,2,1,4)= 0.119424; Wconv(2,3,1,4)= 0.133135;
Wconv(3,1,1,4)= 0.125431; Wconv(3,2,1,4)= 0.108992; Wconv(3,3,1,4)= 0.120018;
Wconv(1,1,2,4)= 0.067385; Wconv(1,2,2,4)= 0.056697; Wconv(1,3,2,4)= 0.061733;
Wconv(2,1,2,4)= 0.075537; Wconv(2,2,2,4)= 0.063899; Wconv(2,3,2,4)= 0.069996;
Wconv(3,1,2,4)= 0.067385; Wconv(3,2,2,4)= 0.056697; Wconv(3,3,2,4)= 0.061733;
Wconv(1,1,3,4)= 0.028461; Wconv(1,2,3,4)= 0.035869; Wconv(1,3,3,4)= 0.028461;
Wconv(2,1,3,4)= 0.035869; Wconv(2,2,3,4)= 0.045010; Wconv(2,3,3,4)= 0.035869;
Wconv(3,1,3,4)= 0.028461; Wconv(3,2,3,4)= 0.035869; Wconv(3,3,3,4)= 0.028461;

bconv = [0.0  0.0  0.0  0.0];
gamma_bn = [1.094532  1.105605  1.088560  1.073892];
beta_bn  = [0.090556  0.056902  0.053351  0.061505];

W_fc = zeros(3, 36);
fc_row1 = [ 0.101473 -0.001327 -0.137586  0.068535  0.308069 -0.148790 ...
           -0.158964  0.273513  0.102334 -0.136562 -0.004401  0.069657 ...
            0.174454 -0.016018 -0.161109  0.138683  0.399052 -0.164786 ...
           -0.186773  0.370254  0.175931 -0.151597 -0.027333  0.140929 ...
            0.101473 -0.001327 -0.137586  0.068535  0.308069 -0.148790 ...
           -0.158964  0.273513  0.102334 -0.136562 -0.004401  0.069657];
fc_row2 = [-0.005699  0.137271 -0.145369  0.017297 -0.179937  0.196277 ...
           -0.038749 -0.148903 -0.136222  0.276876 -0.141999 -0.113413 ...
           -0.047697  0.191723 -0.171017 -0.015892 -0.232492  0.261074 ...
           -0.069083 -0.188778 -0.178631  0.332618 -0.167024 -0.147311 ...
           -0.005699  0.137271 -0.145369  0.017297 -0.179937  0.196277 ...
           -0.038749 -0.148903 -0.136222  0.276876 -0.141999 -0.113413];
fc_row3 = [-0.115774 -0.155944  0.262956 -0.105832 -0.148131 -0.067487 ...
            0.177713 -0.144610  0.013889 -0.160314  0.126400  0.023755 ...
           -0.146757 -0.195704  0.312126 -0.142791 -0.186560 -0.116287 ...
            0.235856 -0.201476 -0.017300 -0.201021  0.174357 -0.013617 ...
           -0.115774 -0.155944  0.262956 -0.105832 -0.148131 -0.067487 ...
            0.177713 -0.144610  0.013889 -0.160314  0.126400  0.023755];
for c = 1:36
    W_fc(1, c) = fc_row1(c);
    W_fc(2, c) = fc_row2(c);
    W_fc(3, c) = fc_row3(c);
end

Xdl = dlarray(batch);
Wconv_dl = dlarray(Wconv);
bconv_dl = dlarray(bconv);
gamma_dl = dlarray(gamma_bn);
beta_dl  = dlarray(beta_bn);
W_fc_dl  = dlarray(W_fc);

Y_c  = conv2d_full(Xdl, Wconv_dl, bconv_dl, 1, 1, 1, 1);
Y_bn = batchnorm(Y_c, gamma_dl, beta_dl);
Y_r  = relu(Y_bn);
Y_p  = avgpool2d(Y_r, 2, 2);
Y_f  = reshape(Y_p, 36, 3);
logits = W_fc_dl * Y_f;
yhat   = softmax(logits, 1);
yp     = extractdata(yhat);

% ===== Step 6: verify each PNG routes to matching class =====
ncorrect = 0;
for n = 1:3
    best = 1; bv = yp(1, n);
    for c = 2:3
        if yp(c, n) > bv, best = c; bv = yp(c, n); end
    end
    fprintf('dl_real_image_pipeline: sample %.0f P=[%.3f %.3f %.3f] -> class %.0f\n', ...
            n, yp(1, n), yp(2, n), yp(3, n), best);
    if best == n, ncorrect = ncorrect + 1; end
end

if ncorrect == 3
    fprintf('dl_real_image_pipeline: PASS (imread -> cat(4) -> CNN end-to-end)\n');
else
    fprintf('dl_real_image_pipeline: FAIL\n');
end
