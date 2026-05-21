# Image Processing Toolbox — examples

Hand-coded Image-Processing-Toolbox subset over the shipped pixel
substrate (no OpenCV / libpng / stb dependency).  Images are `double`
matrices in [0,255] (uint8-class) or [0,1] (double-class) — grayscale M×N
or slice-major M×N×3 truecolor.  See
[`docs/image_toolbox_roadmap.md`](../../docs/image_toolbox_roadmap.md).

## Tier-1 (shipped) — I/O · types · arithmetic · histogram

| Example | User's Guide | Notes |
|---|---|---|
| [`basic_image.m`](basic_image.m) | *Basic Image Import, Processing, and Export* | Build a low-contrast ramp, summarise it (`mean2`/`std2`/`stretchlim`), boost contrast with `imadjust`, then `imwrite` → `imread` round-trip (max diff 0). |

Covered: `imread`/`imwrite` (PGM/PPM/BMP) · `checkerboard` ·
`im2double`/`im2single`/`im2uint8`/`rgb2gray`/`im2gray`/`mat2gray` ·
`imadd`/`imsubtract`/`immultiply`/`imdivide`/`imabsdiff`/`imcomplement`/
`imlincomb` (saturating to [0,255]) · `imhist`/`imadjust`/`stretchlim`/
`mean2`/`std2`.

## Tier-2 (shipped) — spatial filtering + enhancement

| Example | User's Guide | Notes |
|---|---|---|
| [`filtering.m`](filtering.m) | *Designing and Implementing Linear Filters* | `fspecial('gaussian')` + `imfilter` smoothing, salt-and-pepper denoising with `medfilt2`, unsharp `imsharpen`, and `histeq`. |

Covered: `fspecial` (gaussian/average/laplacian/log/sobel/prewitt/disk/
motion) · `imfilter` · `imgaussfilt`/`imboxfilt` · `medfilt2`/`ordfilt2`/
`stdfilt`/`rangefilt` · `histeq`/`adapthisteq` (tiled CLAHE)/`imsharpen`/
`imhistmatch`/`imnoise` (gaussian/salt&pepper/speckle).

## Tier-3 (shipped) — geometric transformations

| Example | User's Guide | Notes |
|---|---|---|
| [`geometric.m`](geometric.m) | *Geometric Transformations* | `imresize`/`imrotate`/`imcrop`, an `affine2d` rotate+scale via `imwarp`, and `fitgeotform2d` recovering a known affine (scale 2 + translate) from matched control points. |

Covered: `imresize` (nearest/bilinear/bicubic) · `imrotate` (`crop`/`loose`)
· `imcrop` · `imtranslate` · `imwarp` (`affine2d`/`projective2d`, auto
bounding box, bilinear inverse-resample) · `imref2d` · `fitgeotform2d`
(least-squares `affine`/`similarity`).  Grayscale + per-channel RGB.
`fliplr`/`flipud`/`rot90` are shipped base ops.

## Tier-4 (shipped) — binarization + morphology + edges

Covered: `graythresh` (Otsu) / `otsuthresh` / `imbinarize` / `im2bw` ·
`strel` (disk/square/rectangle/line) · `imerode`/`imdilate`/`imopen`/
`imclose`/`imtophat`/`imbothat` (grayscale + binary) · `imfill` ('holes') ·
`edge` (Sobel + Canny) · `bwareaopen`.

## Tier-5 (shipped) — segmentation + region analysis

| Example | User's Guide | Notes |
|---|---|---|
| 🎯 [`rice_grains.m`](rice_grains.m) | *Correct Nonuniform Illumination and Analyze Foreground Objects* (the rice.png demo) | **The toolbox headline.** Bright grains over a brightness ramp → `imopen` (disk `strel`) background estimate → `imsubtract` flatten → `imbinarize(graythresh)` → `bwlabel` → `regionprops('Area')`: counts 40 grains and reports mean grain size. |

Covered: `bwlabel` (8-conn) · `regionprops` (`Area`/`Centroid`/
`BoundingBox`/`Perimeter`/`EquivDiameter`/`Extent`/axes/`Eccentricity`/
`Orientation`) · `bwareaopen` · `bweuler` · `label2rgb` · `imsegkmeans`
(reuses the shipped `kmeans`).

## Carve-downs (documented follow-ons)

T1: PNG/JPEG/TIFF decode (PGM/PPM/BMP ship; PNG needs a hand-coded
`inflate`), `imfinfo`, DICOM/HDR, `montage`/`imtile`.  T2: RGB per-channel
`imfilter`, `wiener2`, `entropyfilt`, `imreducehaze`/`locallapfilt`.  T3:
`imwarp` `'OutputView'`/`imref2d` sizing, `rigidtform2d`/`affinetform2d`,
`imresize3`/3-D warp, `normxcorr2`.  See the roadmap.

## Later tiers (planned)

Tier-6 transforms / quality / ROI / colour / block / deblur (`dct2`/
`radon`/`hough` · `psnr`/`ssim` · `poly2mask`/`roifilt2` · `rgb2hsv`/
`rgb2lab` · `blockproc` · `deconvwnr`).  See the roadmap.
