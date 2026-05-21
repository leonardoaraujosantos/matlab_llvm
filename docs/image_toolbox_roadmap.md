# Image Processing Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL**, and **demo** Image-Processing-Toolbox programs.

Source: *Image Processing Toolbox User's Guide* (R2026a, 23 chapters:
Getting Started · Introduction · Reading and Writing Image Data ·
Displaying and Exploring Images · Building GUIs with Modular Tools ·
Geometric Transformations · Image Registration · Designing and
Implementing Linear Filters for Image Data · Image Deblurring ·
Transforms · Morphological Operations · Image Segmentation · Analyze
Images · Image Quality Metrics · ROI-Based Processing · Color · Blocked
Image Processing · Neighborhood and Block Operations · Deep Learning ·
Hyperspectral Image Processing · Optical System Design and Analysis ·
Code Generation · GPU Computing).

This is a **broad, visual, universally-recognised** toolbox — "load an
image, filter it, segment it, measure the objects" is a workflow every
engineer and scientist reaches for. The classical core needs **no Deep
Learning dependency** and is a strong *amplifier* of what the runtime
already ships: the **`uint8` pixel-matrix lane** with saturating
arithmetic (Phase 1.1), `conv2` / `imfilter` / `padarray` / `fft2` /
`imshow`, **3-D arrays** for truecolor/RGB and image stacks, the dense
matrix kernel, and — for `imsegkmeans` — the **`kmeans`** clusterer just
shipped in the Statistics & ML toolbox.

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/images/rice_grains.m`](../examples/images/rice_grains.m): *the
canonical Getting-Started "Correct Nonuniform Illumination and Analyze
Foreground Objects" walkthrough — read a grayscale image of rice grains,
estimate and subtract the uneven background with a morphological opening,
binarize with Otsu's threshold, label the connected components, and
measure each grain with `regionprops`*.  This exercises the
import → arithmetic → filter → morphology → binarize → label → measure
arc end-to-end; achieving it is what closes **Image-Tier-5**.

Companion docs: [`stats_ml_toolbox_roadmap.md`](stats_ml_toolbox_roadmap.md)
(`kmeans` powers `imsegkmeans`; `regionprops` features feed classifiers),
[`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md) (the FFT / filter
kernels overlap), [`plotting.md`](plotting.md) (image display surface),
[`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1**
  is the image-container core: `imread`/`imwrite`/`imfinfo`, the type
  conversions (`im2double`/`im2uint8`/`im2gray`/`rgb2gray`/`mat2gray`),
  and image arithmetic (`imadd`/`imsubtract`/`immultiply`/`imabsdiff`/
  `imcomplement`/`imlincomb`) + `imhist`. **Tier-2** is spatial filtering
  + enhancement (`imfilter`/`fspecial`, `imgaussfilt`/`medfilt2`/
  `ordfilt2`/`imboxfilt`, `histeq`/`adapthisteq`/`imadjust`/`imsharpen`).
  **Tier-3** is geometric transformations (`imresize`/`imrotate`/
  `imcrop`/`imtranslate`/`imwarp` + `affine2d`/`projective2d`/`imref2d`).
  **Tier-4** is morphology + binarization + edges (`imbinarize`/
  `graythresh`/`otsuthresh`, `edge`, `imerode`/`imdilate`/`imopen`/
  `imclose`/`strel`/`bwmorph`/`imfill`/`bwdist`/`watershed`). **Tier-5**
  closes the headline — segmentation + region analysis (`bwlabel`/
  `bwconncomp`/`regionprops`/`bwareaopen`/`bwboundaries`/`label2rgb`/
  `imsegkmeans`/`activecontour`). **Tier-6** is the analysis-and-polish
  layer — transforms (`dct2`/`radon`/`hough`), quality metrics (`psnr`/
  `ssim`/`immse`), ROI (`roipoly`/`poly2mask`/`roifilt2`), colour spaces
  (`rgb2hsv`/`rgb2lab`/`rgb2ycbcr`), block operations (`blockproc`/
  `nlfilter`/`im2col`), and deblurring (`deconvwnr`/`deconvlucy`).
- **Effort** is in the existing Phase 5.6.x cadence (one focused session
  ≈ a half-day; a "week" ≈ 5 sessions). Rough totals: T1 ~3 wk, T2 ~3 wk,
  T3 ~3 wk, T4 ~3.5 wk, T5 ~3.5 wk, T6 ~4 wk (~20 wk full). The single
  **new-infrastructure risk** is image **file I/O** (decoders) — see §1.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
  **Tiers 1–3 cores shipped 2026-05-20** (`runtime/toolbox/images/`) —
  Tier-1 I/O (PGM/PPM/BMP) + types + arithmetic + `imhist`/`imadjust`;
  Tier-2 filtering (`fspecial`/`imgaussfilt`/`medfilt2`/`ordfilt2`/…) +
  enhancement (`histeq`/`adapthisteq`/`imsharpen`/`imnoise`); Tier-3
  geometric (`imresize`/`imrotate`/`imcrop`/`imtranslate`/`imwarp` +
  `affine2d`/`projective2d`/`imref2d` + `fitgeotform2d`).  Tiers 4–6 are
  🔵.  Built on the Tier-0 base (`conv2`/`imfilter`/`padarray`/`fft2`, 3-D
  arrays).
- **Pixel data type**: MATLAB images are `uint8` (0–255), `uint16`,
  `double` (0–1), `single`, or `logical` (binary masks).  The runtime's
  **`uint8` matrix lane with saturating arithmetic ships today** (Phase
  1.1, `matlab_mat_u8`) — the natural representation for `imadd` & co.
  Grayscale = an M×N matrix; truecolor = an M×N×3 3-D array (shipped);
  binary masks = `logical` M×N.  Widening the int lanes (i16/u16) is a
  cheap prerequisite tracked in `feature_status.md`.
- **Object pattern**: spatial-referencing and transform objects
  (`affine2d`, `projective2d`, `imref2d`, `strel`, `offsetstrel`) are
  classdef descriptors — the alloc-then-populate + class-pinned dispatch
  pattern proven by `idss`/`tf`/`mpc`/`ProbDistUnivParam`.  Auto-prepend
  `image_classdefs.m` via the prelude tables.
- **No external dependencies**: matching the project's hand-coded
  precedent — **no OpenCV, no libpng/libjpeg link, no stb vendor.** The
  decoders are hand-coded (PGM/PPM/BMP trivial; PNG via a small inflate;
  baseline JPEG via a compact IDCT decoder as a Tier-1 stretch).

---

## 1. Reusable infrastructure (Tier-0 baseline — no new toolbox code yet)

| Group | Surface (already shipped) | Location | How Image Processing uses it |
|---|---|---|---|
| `uint8` pixel lane | `matlab_mat_u8` + saturating `+ - .* ./`, round-half-away division, `uint8(double)` saturating cast, typed disp/REPL/DAP | `matlab_runtime.cpp` | The native image type; `imadd`/`imsubtract`/`immultiply` are the saturating binops (Tier-1). |
| 3-D arrays | `mat_is_3d` M×N×P descriptors | `matlab_runtime.cpp` | Truecolor M×N×3 RGB + image stacks / volumes. |
| Convolution / filtering | `conv2`, `imfilter`, `padarray`, `fspecial`(check) | `matlab_runtime.cpp` | The Tier-2 linear-filter core; morphology reuses the sliding-window machinery. |
| FFT | `fft2` / `ifft2` (real + complex) | `runtime_complex.cpp` | Frequency-domain filtering, `deconvwnr`, the Tier-6 transforms. |
| Dense linear algebra | `mldivide`, `eig`, `svd`, `inv` | `matlab_runtime.cpp` | `imwarp` affine solves, `regionprops` orientation/eccentricity (2×2 eig), colour-space 3×3 matrices. |
| Clustering | `kmeans` (Lloyd + k-means++) | `runtime/toolbox/stats/runtime_stats.cpp` | **`imsegkmeans`** colour/texture segmentation (Tier-5) is `kmeans` over the pixel feature matrix. |
| Sort / search / reduce | `sort`, `unique`, `histcounts`, `accumarray`, `min`/`max` | `matlab_runtime.cpp` | `imhist`, Otsu threshold search, `regionprops` accumulation, `medfilt2`/`ordfilt2` rank windows, connected-component labelling. |
| Interpolation | `interp1`, `interp2`(check) | `matlab_runtime.cpp` | `imresize`/`imrotate`/`imwarp` resampling kernels (Tier-3). |
| Image display | `imshow` | `runtime/plot/` | Display surface; `imtile`/`montage`/`label2rgb` render through it. |
| Classdef plumbing | `matlab_obj_new`/`_set_*`, class-pinned dispatch, REPL persist, alloc-then-populate | `lib/MLIR/Lowering.cpp` | `affine2d`/`projective2d`/`imref2d`/`strel` descriptors. |

**Net assessment**: the *pixel substrate* (uint8 lane, 3-D arrays,
conv/filter, FFT, interpolation, clustering) is shipped. The genuinely
new code is (a) **image file I/O** — the one new-infrastructure item;
(b) the **morphology engine** (erode/dilate/reconstruct/watershed/bwdist);
(c) **connected-component labelling + `regionprops`**; (d) the
**geometric-resampling** kernels; (e) the **threshold/edge** operators;
and (f) the **colour-space + transform** library.  Each is a self-contained
hand-coded routine over the shipped base.

---

## 2. Tier-1 — Image I/O, types, and arithmetic 🟡 (core shipped)

Goal: get pixels into and out of the workspace and convert between the
image types — the foundation everything else stands on.

**Shipped 2026-05-20** (`runtime/toolbox/images/runtime_images.cpp`):
`imread`/`imwrite` for **PGM/PPM/BMP** (uncompressed real formats) +
`checkerboard` (synthetic); type conversions `im2double`/`im2single`/
`im2uint8`/`rgb2gray`/`im2gray`/`mat2gray`; image arithmetic `imadd`/
`imsubtract`/`immultiply`/`imdivide`/`imabsdiff`/`imcomplement`/
`imlincomb` (saturating to [0,255]); intensity stats `imhist`/`imadjust`
(auto + ranges + gamma)/`stretchlim`/`mean2`/`std2`.  Images are double
matrices ([0,255] uint8-class or [0,1] double-class); RGB is a
slice-major `matlab_mat3`.  **A general compiler fix landed here**: the
shared `pde_table` matcher now materialises single-quoted string literals
(`matlab.const_char`) into `matlab_string*` via `matlab_string_from_literal`,
so any pde_table builtin (`imread('f.pgm')`, `fspecial('gaussian',…)`,
`imnoise(I,'salt & pepper')`) takes a literal filename / option string
directly.  Headline `examples/images/basic_image.m`.  **Tier-1 follow-ons
(🔵):** PNG/JPEG/TIFF decode (hand-coded `inflate` / baseline IDCT —
PGM/PPM/BMP ship), `imfinfo`, DICOM/HDR, `montage`/`imtile`, indexed-image
conversions.

| # | Surface | Algorithm / notes | Runtime entry |
|---|---|---|---|
| 1.1 | `imread` | Decode to a uint8 (or uint16) M×N / M×N×3 array. **PGM/PPM/BMP** (uncompressed, trivial) + **PNG** (hand-coded zlib `inflate` + filters) ship; baseline **JPEG** (IDCT decoder) is a Tier-1 stretch. | `matlab_image_imread` |
| 1.2 | `imwrite` | Encode PGM/PPM/BMP (trivial) + PNG (hand-coded `deflate`, or store-mode chunks). | `matlab_image_imwrite` |
| 1.3 | `imfinfo` | File-header probe → struct (Width/Height/BitDepth/ColorType/FileSize). | `matlab_image_imfinfo` |
| 1.4 | type conversions | `im2double`/`im2single`/`im2uint8`/`im2uint16` (scale + cast), `im2gray`/`rgb2gray` (0.2989R+0.5870G+0.1140B), `mat2gray` (min–max stretch to [0,1]), `gray2ind`/`ind2gray`/`ind2rgb`, `label2rgb`. | `matlab_image_im2*` |
| 1.5 | image arithmetic | `imadd`/`imsubtract`/`immultiply`/`imdivide`/`imabsdiff`/`imcomplement`/`imlincomb` — saturating, over the shipped `uint8` lane. | reuse `matlab_mat_u8` binops |
| 1.6 | intensity stats | `imhist` (256-bin histogram + counts), `mean2`, `std2`, `imadjust` (input/output range remap + gamma), `stretchlim`, `getrangefromclass`. | `matlab_image_imhist` / `_imadjust` |
| 1.7 | display-to-numbers | `imshow` (✅), `imtile`/`montage` (grid composite → array), `imoverlay`. | `runtime/plot/` |

**Headline-within-tier**: UG "Basic Image Import, Processing, and Export" —
`imread` a grayscale image, `imadjust` its contrast, `imwrite` the result;
read it back and confirm `imhist` shifted.

**Compile/Execute wiring**: new `runtime/toolbox/images/runtime_images.cpp`
+ `image_classdefs.m`; register names in `Resolver.cpp`; `pde_table`
loose-match entries in `LowerTensorOps.cpp`; prelude trigger set for
`strel`/`affine2d`/`imref2d`.

---

## 3. Tier-2 — Spatial filtering + enhancement 🟡 (core shipped)

Goal: the linear/nonlinear neighbourhood filters and contrast enhancement
— the most-used image-processing operations.

**Shipped 2026-05-20** (`runtime/toolbox/images/runtime_images.cpp`):
`fspecial` (gaussian / average / laplacian / log / sobel / prewitt / disk
/ motion), `imfilter` (✅), `imgaussfilt`, `imboxfilt`, `medfilt2`,
`ordfilt2`, `stdfilt`, `rangefilt`; enhancement `histeq`, `adapthisteq`
(tiled CLAHE), `imsharpen` (unsharp mask), `imhistmatch`, `imnoise`
(gaussian / salt & pepper / speckle).  Headline
`examples/images/filtering.m`.  **Tier-2 follow-ons (🔵):** RGB
per-channel `imfilter`, `wiener2`, `entropyfilt`, `imreducehaze`/
`locallapfilt`/`imguidedfilter`, `modefilt`.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 2.1 | `imfilter` (✅) + `fspecial` | `fspecial('gaussian'/'average'/'laplacian'/'log'/'sobel'/'prewitt'/'disk'/'motion')` kernel factory; correlation/convolution + replicate/symmetric/circular padding. | `imfilter`, `conv2`, `padarray` |
| 2.2 | Gaussian / box | `imgaussfilt` (separable Gaussian), `imboxfilt` (integral-image box). | separable conv |
| 2.3 | rank / order | `medfilt2` (median), `ordfilt2` (general rank), `modefilt`. | sliding-window sort |
| 2.4 | statistical | `stdfilt`/`rangefilt`/`entropyfilt`, `wiener2` (adaptive). | window reductions |
| 2.5 | histogram enhance | `histeq` (global equalisation), `adapthisteq` (CLAHE — tiled + clip), `imhistmatch`, `imadjust`/`imadjustn`. | `imhist`, cdf |
| 2.6 | sharpening / haze | `imsharpen` (unsharp mask), `imreducehaze`, `locallapfilt`, `imguidedfilter`. | Gaussian, Laplacian |
| 2.7 | noise | `imnoise` (gaussian / salt & pepper / speckle / poisson). | PRNG |

**Headline-within-tier**: UG "Designing and Implementing Linear Filters" —
build a `fspecial('gaussian')`, `imfilter` an image, denoise a
salt-and-pepper image with `medfilt2`, equalise with `histeq`.

---

## 4. Tier-3 — Geometric transformations 🟡 (core shipped)

Goal: resize, rotate, crop, and warp images — resampling with proper
interpolation.

**Shipped 2026-05-20** (`runtime/toolbox/images/runtime_images.cpp` +
`image_classdefs.m`): `imresize` (nearest/bilinear/bicubic-conv; scalar
scale or `[rows cols]`), `imrotate` (`crop`/`loose` bbox), `imcrop`,
`imtranslate`, `imwarp` (affine + projective, auto output bounding box,
bilinear inverse-resampling) with the `affine2d`/`projective2d`/`imref2d`
classdefs (holding the 3×3 forward matrix `T` + `Kind`), and
`fitgeotform2d` (least-squares `affine`/`similarity` from matched control
points → a class-pinned `affine2d`).  All resamplers handle grayscale +
per-channel RGB.  `fliplr`/`flipud`/`rot90` are shipped base ops.  Headline
`examples/images/geometric.m`.  **Tier-3 follow-ons (🔵):** `imwarp`
`'OutputView'`/`imref2d` output sizing, `rigidtform2d`/`affinetform2d`
(premultiply convention), `imresize3`/3-D warp, `maketform`/`imtransform`
(legacy), `normxcorr2`/`fitgeotrans` projective control points.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | `imresize` | Nearest / bilinear / bicubic (antialiased) scaling; scalar scale or `[rows cols]`. | interp kernels |
| 3.2 | `imrotate` | Rotate by an angle, `nearest`/`bilinear`/`bicubic`, `crop`/`loose`. | inverse-map + interp |
| 3.3 | `imcrop` / `imtranslate` | Rectangular crop; sub-pixel translate. | indexing, interp |
| 3.4 | flips | `fliplr`/`flipud`/`rot90`/`flip` (mostly base MATLAB; image-typed). | indexing |
| 3.5 | `imwarp` + tforms | `affine2d`/`affinetform2d`/`projective2d`/`rigidtform2d` classdefs + `imref2d`; inverse-map resampling. | `mldivide`, interp2 |
| 3.6 | `fitgeotform2d` | Estimate a transform from matched control points (`affine`/`projective`/`similarity`) via least squares. | `mldivide` |
| 3.7 | legacy / 3-D | `maketform`/`imtransform`/`tformfwd`; `imresize3`/`imwarp` on volumes. | interp |

**Headline-within-tier**: UG "Geometric Transformations" — build an
`affine2d` rotation+scale, `imwarp` an image into a fixed `imref2d`
output frame, and recover the transform from control points with
`fitgeotform2d`.

---

## 5. Tier-4 — Morphology + binarization + edges 🔵

Goal: the binary/grayscale morphology engine + thresholding + edge
detection — the segmentation prerequisites.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 4.1 | thresholding | `graythresh` (Otsu) + `otsuthresh`, `imbinarize` (global/adaptive), `adaptthresh`, `im2bw`, `multithresh`/`imquantize`. | `imhist`, between-class variance search |
| 4.2 | `edge` | Sobel / Prewitt / Roberts / LoG (zero-cross) / **Canny** (gradient + NMS + hysteresis). | gradient filters |
| 4.3 | `strel` | Structuring elements: `disk`/`square`/`rectangle`/`line`/`diamond`/`octagon`; `offsetstrel`. | classdef |
| 4.4 | binary morphology | `imerode`/`imdilate`/`imopen`/`imclose`/`imtophat`/`imbothat`/`imclearborder`/`imfill` (flood), `bwmorph` (thin/skel/clean/spur/bridge/remove). | sliding window |
| 4.5 | grayscale morphology | `imerode`/`imdilate` on intensity, `imreconstruct` (geodesic), `imhmin`/`imhmax`/`imextendedmax`, `imimposemin`. | reconstruction loop |
| 4.6 | distance / regional | `bwdist` (Euclidean distance transform), `bwperim`, `imregionalmax`/`imregionalmin`, `watershed` (Meyer flooding). | priority flood |

**Headline-within-tier**: UG "Morphological Operations" — `imopen` with a
disk `strel` to clean a binary mask, `imfill` holes, `bwdist` + `watershed`
to split touching objects.

---

## 6. Tier-5 — Segmentation + region analysis (closes the headline) 🔵

Goal: turn pixels into labelled objects and measure them — the analytical
payoff.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | connected components | `bwlabel` / `bwconncomp` (4/8-connectivity union-find), `labelmatrix`, `label2rgb`. | union-find |
| 5.2 | `regionprops` | Area / Centroid / BoundingBox / Perimeter / Eccentricity / Orientation / MajorAxisLength / EquivDiameter / Extent / Solidity / PixelIdxList / … (region accumulation + 2×2 covariance eig). | `accumarray`, 2×2 `eig` |
| 5.3 | region filtering | `bwareaopen`/`bwareafilt`, `bwboundaries` (Moore tracing), `bweuler`, `imclearborder`, `bwselect`. | labelling |
| 5.4 | clustering segmentation | **`imsegkmeans`** (colour/feature k-means) — reuse the shipped `kmeans`; `superpixels` (SLIC), `imsegkmeans3`. | **`kmeans`** |
| 5.5 | region-growing / active | `grayconnected` (flood by intensity), `activecontour` (Chan-Vese / edge), `imseggeodesic`. | iterative loops |
| 5.6 | thresholded measure | `regionprops('table', …)`, `bwpropfilt`. | `regionprops` |

**🎯 Headline (closes Tier-5)**:
[`examples/images/rice_grains.m`](../examples/images/rice_grains.m) —
`imread('rice.png')` → `imopen` with a large disk `strel` to estimate the
uneven background → `imsubtract` → `imbinarize(im, graythresh(im))` →
`bwconncomp`/`bwlabel` → `regionprops(L,'Area','Centroid')` → count the
grains and report mean grain size.  The import → arithmetic → morphology →
binarize → label → measure arc end-to-end.

---

## 7. Tier-6 — Transforms · quality · ROI · colour · block ops 🔵

Goal: the analysis-and-polish layer — frequency/geometric transforms,
quality metrics, ROI processing, colour science, block processing, and
deblurring.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | transforms | `dct2`/`idct2`, `fft2`/`ifft2` (✅), `radon`/`iradon` (filtered back-projection), `hough`/`houghpeaks`/`houghlines`. | FFT, accumulators |
| 6.2 | quality metrics | `immse`, `psnr`, `ssim`/`multissim`, `mean`-based; `niqe`/`brisque` (model-based — carve). | reductions, Gaussian |
| 6.3 | ROI-based | `roipoly`/`poly2mask` (scan-fill), `roifilt2`, `regionfill`, `roicolor`. | mask raster |
| 6.4 | colour spaces | `rgb2hsv`/`hsv2rgb`, `rgb2lab`/`lab2rgb`, `rgb2ycbcr`/`ycbcr2rgb`, `rgb2xyz`, `rgb2ntsc`; `makecform`/`applycform`, `demosaic`. | 3×3 matrices |
| 6.5 | neighborhood/block | `blockproc`, `nlfilter`, `colfilt`, `im2col`/`col2im`, `bwlookup`. | tiling + handle ABI |
| 6.6 | deblurring | `deconvwnr` (Wiener), `deconvlucy` (Lucy-Richardson), `deconvreg`, `edgetaper`, `otf2psf`/`psf2otf`. | FFT |

**Headline-within-tier**: UG "Image Deblurring" — blur an image with a
known PSF, add noise, and restore it with `deconvwnr`; UG "Detecting Lines
Using the Hough Transform" — `edge` → `hough` → `houghlines`.

---

## 8. Carve-outs (explicitly out of scope)

Matching the established roadmap discipline (GUI / Simulink / DL / big-data
deps are always carved):

- **All apps + modular GUI tools** (Image Viewer / `imtool`, Color
  Thresholder, Image Segmenter, Image Region Analyzer, Registration
  Estimator, Volume Viewer, Image Batch Processor, and the Ch.5 modular
  tool builders) — interactive GUI surface; the command-line functions are
  the whole scope here.
- **Deep-learning chapter** (Ch.19: semantic segmentation, denoising /
  super-resolution networks, `imageDatastore`-fed training) — Deep
  Learning Toolbox dependency.
- **Blocked / out-of-core** (Ch.17 `blockedImage`/`bigimage`, MapReduce +
  Hadoop, Image Batch Processor) — out-of-core execution; the in-memory
  array forms are the scope.
- **GPU computing** (Ch.23, `gpuArray`) and **Code Generation** (Ch.22,
  `%#codegen` / GPU Coder) — note the project's *own* emit-c/cpp lanes may
  later cover a subset, but the MathWorks codegen API is carved.
- **Image Registration** (Ch.7) beyond `fitgeotform2d` + `normxcorr2` —
  the intensity-based `imregister`/`imregtform` (mutual-information,
  multimodal, 3-D) optimiser stack is a deferred follow-on.
- **Medical / specialised formats** — DICOM (`dicomread`/`dicominfo`),
  NITF, HDR (`hdrread`), and the full TIFF tag set; basic `imfinfo`/PNG/
  PGM/PPM/BMP cover the mainstream.
- **Hyperspectral** (Ch.20) and **Optical System Design** (Ch.21) — these
  are separate add-on libraries, not core Image Processing.
- **Model-based no-reference quality** (`niqe`/`brisque`) — pretrained
  models; the full-reference metrics (`psnr`/`ssim`/`immse`) ship.
- **Camera calibration / stereo / point clouds** — Computer Vision
  Toolbox surface.

---

## 9. Dependency summary

```
Tier-1 (I/O + types + arithmetic)  ── needs: uint8 lane, 3-D arrays, decoders (NEW), imhist
   ├─ Tier-2 (filtering + enhance)  ── needs: conv2/imfilter/padarray, separable conv, cdf
   ├─ Tier-3 (geometric)            ── needs: interp2, mldivide, affine classdefs
   └─ Tier-4 (morphology + edges)   ── needs: sliding window, flood/reconstruction, Otsu
        └─ Tier-5 (segment + regionprops) ── needs: union-find, accumarray, 2×2 eig, kmeans  ◀── HEADLINE: rice_grains
             └─ Tier-6 (transforms/quality/ROI/colour/block/deblur) ── needs: FFT, 3×3 colour matrices, handle ABI (blockproc)
```

**Critical new build (not reusable from elsewhere)**: (1) the **image
decoders/encoders** (PGM/PPM/BMP trivial; PNG inflate/deflate; baseline
JPEG IDCT as a stretch) — the one genuine new-infrastructure item; (2) the
**morphology engine** (erode/dilate/reconstruct/`bwdist`/`watershed`); (3)
**connected-component labelling + `regionprops`**; (4) the **geometric
resampling** kernels (`imresize`/`imrotate`/`imwarp`); (5) the
**threshold/edge** operators (Otsu, Canny); (6) the **colour-space +
transform** library (`rgb2lab`, `dct2`, `radon`, `hough`). Everything else
(uint8 lane, 3-D arrays, conv/filter, FFT, interpolation, `kmeans`,
classdefs, display) is shipped.

**Sequencing note**: Tier-1 → Tier-4 → Tier-5 is the critical path to the
`rice_grains` headline (import → arithmetic → morphology/binarize →
label/measure). Tiers 2 and 3 are independent and can ship in parallel /
any order. Tier-6 leans on the shipped FFT and on the Tier-2/4/5 building
blocks. **The single risk is image file I/O** — if a hand-coded PNG
decoder proves heavy, Tier-1 can bootstrap on PGM/PPM/BMP (uncompressed,
~50 lines each) plus a synthetic-image generator so Tiers 2–6 are
unblocked while PNG/JPEG land incrementally.
