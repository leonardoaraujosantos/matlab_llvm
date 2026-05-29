# Image Processing Toolbox — Tutorial

A hand-coded Image-Processing-Toolbox subset over the project's pixel substrate — no OpenCV / libpng / libjpeg / stb dependency. Images are `double` matrices in [0,255] (uint8-class) or [0,1] (double-class), grayscale M×N or slice-major M×N×3 truecolor. Notably, `imread` decodes **real PNG and baseline JPEG** files (and PGM/PPM/BMP), all from hand-coded codecs.

## Supported features

- **I/O, types, arithmetic, histogram:** `imread` (PGM/PPM/BMP + real PNG + baseline JPEG), `imwrite` (PGM/PPM/BMP + lossless PNG), `checkerboard`, `im2double` / `im2uint8` / `rgb2gray` / `im2gray` / `mat2gray`, `imadd` / `imsubtract` / `immultiply` / `imdivide` / `imabsdiff` / `imcomplement` / `imlincomb` (saturating), `imhist` / `imadjust` / `stretchlim` / `mean2` / `std2`.
- **Filtering & enhancement:** `fspecial`, `imfilter`, `imgaussfilt` / `imboxfilt`, `medfilt2` / `ordfilt2` / `stdfilt` / `rangefilt`, `histeq` / `adapthisteq` (CLAHE), `imsharpen`, `imhistmatch`, `imnoise`.
- **Geometric:** `imresize` (nearest/bilinear/bicubic), `imrotate`, `imcrop`, `imtranslate`, `imwarp` with `affine2d` / `projective2d`, `imref2d`, `fitgeotform2d`.
- **Binarize / morphology / edges:** `graythresh` (Otsu) / `otsuthresh` / `imbinarize` / `im2bw`, `strel`, `imerode` / `imdilate` / `imopen` / `imclose` / `imtophat` / `imbothat`, `imfill`, `edge` (Sobel + Canny), `bwareaopen`.
- **Segmentation / region analysis:** `bwlabel` (8-conn), `regionprops`, `bweuler`, `label2rgb`, `imsegkmeans`.
- **Transforms / quality / colour:** `dct2` / `idct2`, `radon`, `hough` / `houghpeaks`, `psnr` / `ssim` / `immse`, `rgb2hsv`/`hsv2rgb`, `rgb2ycbcr`/`ycbcr2rgb`, `rgb2lab`/`lab2rgb`, `deconvwnr`, plus 3-D indexing `rgb(:,:,k)` and `cat(3,...)`.

## Build & run

```bash
build/matlabc -emit-llvm examples/images/rice_grains.m > /tmp/rice_grains.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/rice_grains.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/rice_grains
/tmp/rice_grains
```

Swap `rice_grains` for any other file under `examples/images/`.

## Worked examples

### Count rice grains under uneven illumination — HEADLINE  (`examples/images/rice_grains.m`)

The Getting-Started "Correct Nonuniform Illumination and Analyze Foreground Objects" workflow (the MATLAB `rice.png` demo): morphological background estimate → subtract to flatten → Otsu threshold → label → measure.

```matlab
% ----- 1-2. flatten the illumination by background subtraction ---------
bg   = imopen(I, strel('disk', 12));            % opening removes the grains
flat = imsubtract(I, bg);

% ----- 3. threshold to a binary mask ----------------------------------
level = graythresh(flat);
BW    = imbinarize(flat, level);
BW    = bwareaopen(BW, 8);                       % drop specks

% ----- 4. label + measure ---------------------------------------------
L = bwlabel(BW);
n = max(max(L));
areas = regionprops(L, 'Area');
fprintf('grains detected = %.0f\n', n);
fprintf('mean grain area = %.1f px\n', mean(areas));
```

A morphological opening with a disk `strel` larger than the grains estimates the diagonal brightness ramp; subtracting it flattens the field. `graythresh` (Otsu) picks the binarization level, `bwareaopen` removes specks, `bwlabel` does an 8-connected component labeling, and `regionprops(L,'Area')` returns the per-grain areas. The run detects the planted grain field and reports its mean size.

### Real PNG round-trip  (`examples/images/read_write_png.m`)

`imwrite` produces standard lossless PNG and `imread` decodes it bit-exact — then the loaded image is analysed as if loaded from disk.

```matlab
photo = cat(3, R, G, B);
imwrite(photo, '/tmp/demo_photo.png');
loaded = imread('/tmp/demo_photo.png');
fprintf('round-trip max channel error = %.0f (lossless)\n', ...
        max(max(imabsdiff(loaded(:,:,1), photo(:,:,1)))));
gray = rgb2gray(loaded);
bw   = imbinarize(gray);
```

The PNG codec (deflate/inflate + CRC32/adler32) is hand-coded with no zlib. Baseline JPEG (`*.jpg`) decode works the same way, but is lossy so its round-trip is not bit-exact.

### Linear filtering & enhancement  (`examples/images/filtering.m`)

```matlab
g = fspecial('gaussian', 7, 1.5);
smooth = imfilter(I, g);
noisy = imnoise(I, 'salt & pepper', 0.15);
clean = medfilt2(noisy, [3 3]);
sharp = imsharpen(I);
eq = histeq(I);
```

`fspecial` builds the kernel (it sums to 1.0 for the Gaussian), `imfilter` convolves, `medfilt2` removes salt-and-pepper noise, and `histeq` equalises the histogram to the full range.

### Geometric transforms & control-point recovery  (`examples/images/geometric.m`)

```matlab
half = imresize(I, 0.5);                          % bicubic by default
rot  = imrotate(I, 30, 'bilinear');               % 'loose' bounding box
A    = affine2d([s*cos(th) s*sin(th) 0; -s*sin(th) s*cos(th) 0; 0 0 1]);
warped = imwarp(I, A);
tform  = fitgeotform2d(moving, fixed, 'affine');  % recover from matched points
```

`imwarp` inverts the 3×3 transform and bilinearly resamples; `fitgeotform2d` recovers a known affine (scale 2 + translate) from four matched control points by least squares.

### Other examples (briefly)

- `basic_image.m` — `stretchlim`/`imadjust` contrast boost and an `imwrite`→`imread` PGM round-trip.
- `transforms.m` — `dct2`/`idct2` energy compaction, HSV/YCbCr/Lab colour round-trips, `hough`/`houghpeaks` line detection, `psnr`/`ssim` metrics, and `deconvwnr` Wiener deblur.
- `channel_split.m` — build truecolor with `cat(3,...)`, split with `rgb(:,:,k)`, boost a channel with `imadd`, and merge back.

## Limitations & carve-outs

- **All apps / GUI tools** (Image Viewer, Color Thresholder, Image Segmenter, Registration Estimator, …) — command-line functions only.
- **Deep-learning chapter** (semantic segmentation, denoising/super-resolution nets) — Deep Learning Toolbox dependency.
- **Blocked / out-of-core** (`blockedImage`/`bigimage`, MapReduce) and **GPU / GPU-Coder** — in-memory arrays only.
- **Image registration beyond `fitgeotform2d`** (`imregister`/`imregtform`, mutual-information / 3-D) — deferred.
- **Medical / specialised formats** (DICOM, NITF, HDR, full TIFF tag set) — PNG/PGM/PPM/BMP cover the mainstream.
- **Hyperspectral / optical design**, **model-based no-reference quality** (`niqe`/`brisque`), and **camera-calibration / stereo / point clouds** (Computer Vision Toolbox) — out of scope.

## See also

- Roadmap: [`image_toolbox_roadmap.md`](../image_toolbox_roadmap.md)
- Examples: `examples/images/`
