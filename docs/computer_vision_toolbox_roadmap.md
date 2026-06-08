# Computer Vision Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ Plot) needs to ship in order to faithfully **compile and execute**,
**debug/REPL/JIT**, and **demo** Computer-Vision-Toolbox programs.

Source: *Computer Vision Toolbox User's Guide* (R2026a, ~2144 pp, 25
chapters: Camera Calibration & SfM · Code Generation · Image/Video
Classification · Instance Segmentation · Keypoint Detection · Object
Detection · Semantic Segmentation · Automated Visual Inspection · Feature
Detection & Extraction · Ground-Truth Labeling · Lidar/Point Cloud ·
Simulink · Text Detection/OCR · Tracking & Motion Estimation · Faster
R-CNN · Labelers · Featured Examples · SfM & Visual SLAM · Point Cloud
Processing · Installer · OpenCV Interface · I/O & Conversions · Display &
Graphics · Registration & Stereo Vision · Object Detection).

This is a **direct, high-leverage extension of the already-shipped Image
Processing Toolbox** — the same posture as the recent Bioinformatics
(over Stats) and Bluetooth (over Comm/DSP) roadmaps. The Computer Vision
*classical core* is "detect features in an image, describe them, match
them across images, estimate the geometric transform, and warp/measure" —
and every primitive that workflow needs is already in the tree (verified
in `lib/Sema/Resolver.cpp` + `runtime/toolbox/images/runtime_images.cpp`):

- **Image substrate** — the `uint8`/`double` pixel-matrix lane, 3-D RGB
  arrays, `imread`/`imwrite` (real PNG/JPEG codecs), `rgb2gray`/`im2gray`,
  `imfilter`/`conv2`/`fspecial`, `imgradient`/`edge`, `imresize`,
  morphology, `regionprops`, `hough` — all ✅ shipped.
- **Geometric transforms** — `affine2d`/`projective2d`/`imref2d`,
  `imwarp`, `fitgeotform2d` (LS transform fit) ✅ shipped. CV's
  `estgeotform2d`/`estimateGeometricTransform2D` is `fitgeotform2d`
  wrapped in a RANSAC loop over matched points.
- **Dense linear algebra** — `svd`/`qr`/`pinv`/`mldivide`/`eig` ✅: the
  homography/fundamental/essential-matrix DLT solves, triangulation, PnP,
  and the HOG/feature math are all linear algebra over the shipped kernel.
- **Stats / ML** — `kmeans` (the bag-of-features visual vocabulary) and
  `fitcsvm`/`fitcecoc` (the image-category classifier) ✅ shipped.
- **Deep Learning** — `dlnet` ✅ shipped; the DL object detectors /
  segmenters can run *inference* on it, but they need trained weights +
  the training loop, so the DL chapters are a carve-down (see §9).
- **Plotting** — Cairo `imshow`/`imagesc`/`plot`/`line` ✅ for
  `insertShape`/`insertText`/feature-overlay display.

The classical core needs **no Deep Learning dependency and no external
library** (no OpenCV) — every detector/descriptor/estimator is a
hand-coded routine over the shipped image + linear-algebra kernels, and
every constant (HOG cell geometry, BRISK/ORB sampling patterns, the
RANSAC thresholds) is baked in.

The headline tracer-bullet (the gating example for the whole roadmap) is
[`examples/vision/feature_match_panorama.m`](../examples/vision/feature_match_panorama.m):
*the canonical "Create Panorama" / "Find Object in Cluttered Scene"
workflow — detect corner features in two overlapping images
(`detectHarrisFeatures`), describe them (`extractFeatures`), match them
(`matchFeatures`), robustly estimate the geometric transform between them
(`estgeotform2d` with RANSAC), and warp one onto the other's frame with
the shipped `imwarp`*. This exercises the
detect → extract → match → estimate → warp arc end-to-end; achieving it
closes **CV-Tier-1/2** (feature matching + geometric registration — the
single most common reason anyone reaches for the classical toolbox).
Companion tracer-bullets:
[`examples/vision/optical_flow_motion.m`](../examples/vision/optical_flow_motion.m)
(Lucas-Kanade optical flow, **CV-Tier-4**) and
[`examples/vision/stereo_depth.m`](../examples/vision/stereo_depth.m)
(disparity + triangulation, **CV-Tier-5**).

Companion docs:
[`image_toolbox_roadmap.md`](image_toolbox_roadmap.md) (the pixel-matrix /
`imfilter` / `imwarp` / `regionprops` / `edge` / `hough` substrate — CV is
its natural extension), [`stats_ml_toolbox_roadmap.md`](stats_ml_toolbox_roadmap.md)
(`kmeans` powers `bagOfFeatures`, `fitcecoc`/`fitcsvm` the category
classifier), [`deep_learning_toolbox_roadmap.md`](deep_learning_toolbox_roadmap.md)
(the carved DL-detector inference path), [`sensor_fusion_toolbox_roadmap.md`](sensor_fusion_toolbox_roadmap.md)
(the `vision.KalmanFilter` / `assignDetectionsToTracks` tracking tier
reuses the shipped tracking-filter + Munkres assignment surface),
[`plotting.md`](plotting.md), [`feature_status.md`](feature_status.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. **Tier-1** is
  feature detection + description + matching (`detectHarrisFeatures` /
  `detectFASTFeatures` / `extractFeatures` / `matchFeatures` /
  `extractHOGFeatures`) — the classical-CV foundation. **Tier-2** is
  geometric-transform estimation + image registration (`estgeotform2d` /
  `estimateGeometricTransform2D` via RANSAC, panorama stitching).
  **Tier-3** is bounding-box utilities + annotation (`bboxOverlapRatio` /
  `selectStrongestBbox` / `insertShape`/`insertText`/`insertMarker` /
  `insertObjectAnnotation`). **Tier-4** is optical flow + video motion
  (`opticalFlowLK` / `opticalFlowHS` / `opticalFlowFarneback`). **Tier-5**
  is camera geometry + stereo (`cameraIntrinsics` / `undistortImage` /
  `triangulate` / `estimateFundamentalMatrix` / `disparityBM` /
  `reconstructScene`). **Tier-6** is the application layer — point clouds
  (`pointCloud` / `pcread` / `pcdownsample` / `pcfitplane` /
  `pcregistericp`), bag-of-features image classification (over Stats), and
  `ocr`.
- **Effort** is in the existing Phase 5.6.x cadence (one focused session ≈
  a half-day; a "week" ≈ 5 sessions). Rough totals: **T1 ~2.5 wk · T2 ~1.5
  wk · T3 ~1 wk · T4 ~2 wk · T5 ~2.5 wk · T6 ~2.5 wk (~12 wk full)**. Each
  tier is independently shippable and demoable; **T1 + T2 (~4 wk) close the
  80% feature-matching / registration workflow** — the most common reason
  anyone reaches for the classical toolbox. Badge would advance by one.
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started. **ALL 6 TIER
  CORES SHIPPED 2026-06-08 (Phases A+B+C)** in
  [`runtime/toolbox/vision/runtime_vision.cpp`](../runtime/toolbox/vision/runtime_vision.cpp)
  (~700 LOC, self-contained over the shipped image-gradient + dense-linalg
  substrate). T1 `detectHarrisFeatures`/`detectMinEigenFeatures`/
  `detectFASTFeatures`/`extractFeatures`(patch)/`matchFeatures`/
  `extractHOGFeatures`/`extractLBPFeatures` · T2 `estgeotform2d`(RANSAC)/
  `estimateGeometricTransform2D`/`estimateFundamentalMatrix` · T3
  `bboxOverlapRatio`/`selectStrongestBbox`/`bbox2points`/`insertShape`/
  `insertMarker` · T4 `opticalFlowLK`/`opticalFlowHS` · T5 `triangulate`/
  `disparityBM` · T6 `pcread`/`pcwrite`/`pcdownsample`/`pcfitplane`/
  `pcregistericp`. 4 gating tests
  (`test/Run/vision_{features,geotrans,bbox_flow,stereo_cloud}.m`) + 4
  examples (`examples/vision/`). Suite: **Run 758/0, frontend 83/0,
  emit-c/py/ts 324/266/231 /0, JIT/DAP gate OK, examples-sweep 0
  regressions** — every example runs identically under AOT and the
  JIT-interpreted (`-dap`) path.
  **Implementation notes / deviations from the planned API** (documented
  carve-downs — the numeric workflow is faithful, the surface is simplified
  to the robust function lane):
  - **Function forms returning plain real matrices, not classdefs**: feature
    detectors return a `K×2 [x y]` location matrix (strongest-first); no
    `cornerPoints`/`pointCloud`/`opticalFlow` objects — point clouds are
    `N×3` matrices, optical flow is `[Vx; Vy]` stacked vertically (`2M×N`).
    This sidesteps JIT classdef-state and reuses the proven spec-table
    wiring (the Bluetooth precedent). The object surface is a faithful-API
    follow-on. `estgeotform2d` returns a `3×3` matrix → wrap in the shipped
    `affine2d(T)` then `imwarp` (verified end-to-end).
  - **`extractFeatures` = normalized intensity-patch (11×11) descriptors**,
    matched by SSD + Lowe ratio test — translation/mild-transform robust
    (recovers shifts exactly); rotation-invariant BRISK/ORB descriptors are
    a refinement (the `detectORBFeatures` slot is deferred).
  - **RANSAC + ICP determinism**: `estgeotform2d`/`pcfitplane` use a fixed
    internal LCG (reproducible regardless of prior `rng`); recovers known
    transforms exactly + rejects gross outliers. `pcregistericp` is
    point-to-point Kabsch ICP — sensitive to large off-model outliers (no
    inlier rejection yet), so register inlier sets.
  - **JIT/DAP traps respected**: examples use the `for i = 1:N` range form
    (`for x = vec` does not lower under ReplMode); shifts via `imtranslate`;
    no `circshift`/`nnz`/`zeros(size(..))`/1-arg-`zeros` (unsupported);
    error counts via `round`/`abs` not raw comparison results.
  The reuse anchors (`imfilter`/`imwarp`/`fitgeotform2d`/`affine2d`/`edge`/
  `hough`/`rgb2gray`/`imgaussfilt`/`imtranslate`, `svd`/`qr`/`pinv`,
  `kmeans`/`fitcsvm`) are all ✅ shipped.
- **Images are pixel matrices; feature points + boxes are numeric
  matrices** — the exact lanes already in use. A grayscale image is `M×N`
  `double`/`uint8`; an RGB image is `M×N×3`; a set of feature points is an
  `K×2` `[x y]` location matrix (+ a parallel metric/scale/orientation
  column); descriptors are a `K×D` matrix; bounding boxes are `K×4`
  `[x y w h]`. No new container type for the numeric core — CV rides the
  shipped real-matrix + 3-D-array lanes.
- **Feature-point + camera objects use the shipped classdef recipe** —
  `cornerPoints`/`SURFPoints`/`ORBPoints`, `cameraIntrinsics`/
  `cameraParameters`, `affine2d`/`projective2d` (already shipped),
  `pointCloud`, `opticalFlow`, `binaryFeatures` are property-holder
  classdefs (the `phytree`/`DataMatrix`/`affine2d` alloc-then-populate +
  class-pinned-dispatch pattern, auto-prepended via `vision_classdefs.m`).
  Where state-mutation under the JIT is awkward, the Bluetooth precedent
  applies: ship the **function form** (`detectHarrisFeatures` returns a
  plain `K×2` location matrix + metric column; the `cornerPoints` object
  is a faithful-API follow-on) and document the deviation.
- **CV constants are baked-in tables** — HOG cell/block geometry, the
  BRISK/ORB sampling-pattern offsets, the optical-flow kernels, the RANSAC
  default thresholds — all static arrays in the runtime (the Image
  `fspecial` / Comm 5G-NR-base-matrix / Bluetooth channel-map precedent).
- **The Deep-Learning detectors, apps, Simulink, OpenCV interface, and
  Visual SLAM are carved out** (see §9): YOLO/SSD/R-CNN/Mask-R-CNN/SOLOv2
  object detection + semantic/instance/keypoint segmentation + video
  classification + ReID/DeepSORT need trained networks + the training loop;
  the Camera Calibrator / Image-Video Labeler / Ground-Truth apps, the
  Simulink block library, the `mexOpenCV` interface, NeRF, and the
  monocular/stereo Visual-SLAM pipelines are all out of scope. The
  classical algorithms those examples also use (feature matching, geometric
  estimation, triangulation) **are** in scope via Tiers 1/2/5.

---

## 1. Reusable infrastructure (Tier-0 baseline — no Computer Vision code yet)

| Group | Surface (already shipped) | Location | How Computer Vision uses it |
|---|---|---|---|
| Pixel-matrix lane | `uint8`/`double` images, 3-D RGB arrays, `im2double`/`im2gray`/`rgb2gray` | `runtime/toolbox/images/runtime_images.cpp` ✅ | Every image input/output; grayscale conversion before feature detection (Tier-1/4/5). |
| Image filtering / gradients | `imfilter`, `conv2`, `fspecial`, `imgradient`, `edge` | `runtime/toolbox/images/` ✅ | Harris/FAST corner response, HOG gradient histograms, optical-flow spatial/temporal derivatives (Tier-1/4). |
| Image I/O codecs | `imread`/`imwrite` (real PNG/JPEG) | `runtime/toolbox/images/` ✅ | Loading the demo image pairs; no bundled-binary dependency (Tier-1/2). |
| Geometric transforms | `affine2d`/`projective2d`/`imref2d`, `imwarp`, `fitgeotform2d` | `runtime/toolbox/images/` ✅ | `estgeotform2d` = `fitgeotform2d` in a RANSAC loop; `imwarp` does the panorama/rectification warp (Tier-2/5). |
| Dense linear algebra | `svd`, `qr`, `pinv`, `mldivide`, `eig`, `det` | `runtime/matlab_runtime.cpp` ✅ | Homography/fundamental/essential DLT, triangulation, PnP, HOG block normalization (Tier-2/5). |
| Hough / shape | `hough`, `houghpeaks`, `regionprops`, `bwlabel` | `runtime/toolbox/images/` ✅ | Line/shape detection; MSER region props; connected-component features (Tier-1/3). |
| Stats / ML | `kmeans`, `fitcsvm`, `fitcecoc`, `pdist2`, `predict` | `runtime/toolbox/stats/runtime_stats.cpp` ✅ | `bagOfFeatures` visual vocabulary (`kmeans`); `trainImageCategoryClassifier` (`fitcecoc`); descriptor matching distances (Tier-1/6). |
| Tracking filters / assignment | `trackingKF`, `assignmunkres` | `runtime/toolbox/fusion/runtime_fusion.cpp` ✅ | `vision.KalmanFilter` + `assignDetectionsToTracks` reuse the shipped KF + Munkres surface (Tier-3 tracking follow-on). |
| Deep Learning | `dlnetwork`, conv/activation inference | `runtime/toolbox/dlnet/runtime_dlnet.cpp` ✅ | Inference path for the carved DL detectors/segmenters (§9) — not in the classical core. |
| Classdef plumbing | `matlab_obj_new`/`_set_*`/`_get_mat`, kwarg-ctor, class-pinned dispatch, REPL persist, DAP render | `lib/MLIR/Lowering.cpp`, `runtime/runtime_debug.cpp` | `cornerPoints`/`cameraIntrinsics`/`pointCloud`/`opticalFlow` objects (Tier-1/5/6). |
| Name/value + option strings | option-string read in runtime (`fspecial`/`nwalign` path) | `lib/MLIR/Passes/LowerTensorOps.cpp` | `matchFeatures(...,'MaxRatio',0.6,'Method','Exhaustive')`, `estgeotform2d(...,'projective')` (Tier-1/2). |
| Plotting | Cairo `imshow`/`imagesc`/`plot`/`line`/`rectangle` | `runtime/plot/` | `insertShape`/`insertText`/`insertMarker` overlays, feature/match display, optical-flow quiver (Tier-1/3/4). |

**Net assessment**: the *image + linear-algebra + ML substrate* (pixel
lanes, filtering/gradients, codecs, geometric warp, SVD/QR/pinv, `kmeans`/
`fitcsvm`, classdef + plotting) is **already shipped**. The genuinely new
code is (a) the **feature detectors** (Harris/FAST/Shi-Tomasi corner
response over the shipped gradients; BRISK/ORB sampling patterns), (b) the
**descriptors + matcher** (HOG/LBP/binary descriptors + `matchFeatures`
nearest-neighbour-with-ratio-test), (c) the **RANSAC geometric estimator**
(`estgeotform2d` looping `fitgeotform2d`), (d) the **optical-flow solvers**
(Lucas-Kanade / Horn-Schunck / Farneback), (e) the **camera-geometry +
stereo** layer (fundamental/essential matrix, triangulation, block-matching
disparity), and (f) the **point-cloud + bag-of-features + OCR** application
layer. Each is a self-contained hand-coded routine over the shipped base —
the heavy lifting (filtering, SVD, warp, clustering) is done.

---

## 2. CV-Tier-1 — Feature detection + description + matching ✅

Goal: find salient points in an image, describe their local appearance,
and match them across images — the classical-CV foundation everything else
builds on. Closes half the headline.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 1.1 | `detectHarrisFeatures` / `detectMinEigenFeatures` | Harris / Shi-Tomasi corner response from the structure tensor (gradient products smoothed by a Gaussian); non-max suppression; return `K×2` `[x y]` + metric. | `imgradient`, `imfilter` |
| 1.2 | `detectFASTFeatures` | FAST corner test (Bresenham circle of 16, contiguous-arc threshold); NMS. | pixel access |
| 1.3 | `detectBRISKFeatures` / `detectORBFeatures` | Scale-space FAST + orientation; binary descriptor sampling pattern. *(stretch within tier)* | 1.2, baked pattern |
| 1.4 | `cornerPoints` / `SURFPoints` / `ORBPoints` (classdef) | Point object carrying `Location`/`Metric`/`Scale`/`Orientation` + `selectStrongest`/`length`/plot. Function form returns the bare matrices (the Bluetooth precedent). | classdef |
| 1.5 | `extractFeatures` | Descriptor extraction at points: HOG-patch / upright-BRISK binary / intensity-patch; returns `K×D` features + valid points. | `imfilter`, gradients |
| 1.6 | `extractHOGFeatures` | Histogram-of-oriented-gradients over a cell/block grid with block normalization; returns the feature vector (+ optional visualization). | `imgradient`, `svd`-free L2 norm |
| 1.7 | `extractLBPFeatures` | Local binary patterns histogram. | pixel access |
| 1.8 | `matchFeatures` | Nearest-neighbour descriptor matching (SSD / Hamming) with Lowe ratio test + max-distance; returns index pairs. | `pdist2`, sort |
| 1.9 | display | `imshow` + `plot`/`insertMarker` of points; `showMatchedFeatures` (two images side-by-side with match lines). | `runtime/plot/` |

**Headline-within-tier**: detect + match — `p1=detectHarrisFeatures(I1);
p2=detectHarrisFeatures(I2); [f1,v1]=extractFeatures(I1,p1); ...;
idx=matchFeatures(f1,f2)` returns a plausible set of correspondences.

**Compile/Execute wiring**: new `runtime/toolbox/vision/runtime_vision.cpp`;
register the detector/extractor/matcher names in `Resolver.cpp`; feature
points + descriptors are plain matrices (the shipped real-matrix lane);
`extractFeatures` is a 2-output builtin (`[features, validPoints]`) via the
multi-output splitter; option strings read in the runtime.

**REPL/JIT + Debug**: matrices render in the REPL/DAP panes already; if the
`cornerPoints` classdef is used, mind the recurring **ReplMode workspace
round-trip** trap (see the Bluetooth/Bioinformatics fixes).

---

## 3. CV-Tier-2 — Geometric-transform estimation + image registration ✅

Goal: robustly fit the transform between two matched point sets and warp —
the panorama / object-location / image-alignment payoff. Closes the
headline.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 2.1 | `estgeotform2d` / `estimateGeometricTransform2D` | RANSAC loop: sample minimal point sets, fit a `rigid`/`similarity`/`affine`/`projective` transform with the shipped `fitgeotform2d`, score inliers, refit on the consensus set. Returns an `affine2d`/`projective2d` + inlier mask. | `fitgeotform2d`, RANSAC, PRNG |
| 2.2 | `estimateFundamentalMatrix` | 8-point / normalized-8-point DLT (+ RANSAC/MSAC) → `3×3` F. | `svd` |
| 2.3 | `estimateEssentialMatrix` | From F + camera intrinsics, or normalized-coordinates 5/8-point. | `svd`, 2.2 |
| 2.4 | panorama / registration | Chain `matchFeatures`→`estgeotform2d`→`imwarp` into a common frame; blend by max/average. | `imwarp` (Image) |
| 2.5 | `bbox2points` / transform application | Map box corners through the estimated transform (object localization). | matrix mult |

**Headline-within-tier (whole-roadmap tracer-bullet)**:
`feature_match_panorama.m` — two overlapping images → detect/extract/match
→ `estgeotform2d` (RANSAC) → `imwarp` → stitched panorama; report the inlier
count and the recovered rotation/scale.

**Compile/Execute wiring**: `estgeotform2d` returns a class-pinned
`affine2d`/`projective2d` (already-shipped classdefs) + an inlier-index
output (multi-return); the RANSAC PRNG is seeded for deterministic tests;
`imwarp` is the shipped Image runtime.

---

## 4. CV-Tier-3 — Bounding boxes + annotation + classical detection ✅

Goal: the detection-support utilities + drawing-on-images surface that
every detection/tracking example uses.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 3.1 | `bboxOverlapRatio` / `bboxIntersectionOverUnion` | IoU / overlap matrix between two box sets. | matrix ops |
| 3.2 | `selectStrongestBbox` / `selectStrongestBboxMulticlass` | Greedy non-max suppression by score + overlap threshold. | 3.1, sort |
| 3.3 | `bbox2points` / `bboxresize` / `bboxcrop` | Box ↔ corner conversion, resize/crop with an image. | matrix ops |
| 3.4 | `insertShape` / `insertMarker` / `insertText` / `insertObjectAnnotation` | Draw rectangles/lines/circles/markers/labels onto an image matrix (returns the annotated image). | `runtime/plot/` raster or direct pixel writes |
| 3.5 | `vision.CascadeObjectDetector` (Viola-Jones) | Cascade detector over an integral-image Haar feature evaluation; ships with the bundled face/eye cascade tables. *(stretch — large baked cascade)* | integral image |

**Headline-within-tier**: `boxes = [...]; iou = bboxOverlapRatio(boxes,
boxes); kept = selectStrongestBbox(boxes, scores); J =
insertObjectAnnotation(I,'rectangle',boxes(kept,:),labels)`.

**Compile/Execute wiring**: bbox utilities are matrix-in/matrix-out
builtins; `insert*` functions write into a copy of the pixel matrix (or via
the Cairo raster path) and return the annotated image; the Cascade detector
(if shipped) reads a baked Haar-cascade table.

---

## 5. CV-Tier-4 — Optical flow + video motion ✅

Goal: estimate per-pixel / sparse motion between frames.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 4.1 | `opticalFlowLK` + `estimateFlow` | Lucas-Kanade: solve the 2×2 windowed normal equations per pixel from the spatial/temporal gradients. | `imgradient`, 2×2 solve |
| 4.2 | `opticalFlowHS` | Horn-Schunck: iterative global smoothness-regularized flow. | gradients, Jacobi iter |
| 4.3 | `opticalFlowFarneback` | Polynomial-expansion dense flow. *(stretch within tier)* | gradients, `mldivide` |
| 4.4 | `opticalFlow` object + quiver display | Flow object carrying `Vx`/`Vy`/`Magnitude`/`Orientation`; `plot` as a quiver overlay. | classdef, `runtime/plot/` |
| 4.5 | `vision.PointTracker` (KLT) | Pyramidal Lucas-Kanade point tracking across frames. *(stretch)* | 4.1, `imresize` |

**Headline-within-tier (Tier-4 tracer-bullet)**: `optical_flow_motion.m` —
two frames of a moving object → `opticalFlowLK` → report the dominant
flow direction/magnitude + a quiver overlay.

**Compile/Execute wiring**: flow fields are `M×N` matrices (or an
`opticalFlow` classdef carrying `Vx`/`Vy`); the per-pixel 2×2 solve is a
hand-coded runtime routine over the shipped gradient kernels.

---

## 6. CV-Tier-5 — Camera geometry + stereo vision ✅

Goal: the calibrated-camera geometry that turns pixels into 3-D — the
basis of stereo depth, SfM, and measurement.

| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 5.1 | `cameraIntrinsics` / `cameraParameters` (classdef) | Intrinsic matrix `K` + radial/tangential distortion coefficients; `pixelsToWorld`/`worldToImage` projection. | classdef |
| 5.2 | `undistortImage` / `undistortPoints` | Apply the inverse distortion model + remap (reuses `imwarp`-style sampling). | `imwarp`, model eval |
| 5.3 | `triangulate` | Linear DLT triangulation of a 3-D point from two camera projection matrices + image correspondences. | `svd` |
| 5.4 | `estimateCameraProjection` / PnP | Camera matrix from 3-D↔2-D correspondences (DLT). | `svd` |
| 5.5 | `rectifyStereoImages` | Rectify a stereo pair to row-aligned epipolar geometry. | 2.2, `imwarp` |
| 5.6 | `disparityBM` / `disparitySGM` | Block-matching / semi-global disparity map from a rectified pair. | windowed SSD, DP |
| 5.7 | `reconstructScene` | Disparity + stereo params → dense 3-D point cloud. | 5.3, 5.6 |

**Headline-within-tier (Tier-5 tracer-bullet)**: `stereo_depth.m` —
a rectified stereo pair → `disparityBM` → `reconstructScene` (or
`triangulate` on matched points) → report a depth at a chosen pixel.

**Compile/Execute wiring**: `cameraIntrinsics`/`cameraParameters` are
classdefs; `triangulate`/`estimateFundamentalMatrix` are SVD over the
shipped kernel; disparity is a hand-coded windowed-cost routine; the
reconstructed point set is an `N×3` matrix (or `pointCloud`, Tier-6).

---

## 7. CV-Tier-6 — Application layer: point clouds + bag-of-features + OCR ✅

Goal: the high-reuse application surface — 3-D point clouds, image-category
classification over Stats, and text recognition.

### 7a. Point cloud
| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.1 | `pointCloud` (classdef) | `Location` (`N×3`) + optional `Color`/`Normal`/`Intensity`; `findNearestNeighbors`/`select`. | classdef, `pdist2` |
| 6.2 | `pcread` / `pcwrite` | PLY (ASCII + binary) read/write — a hand-coded parser (the FASTA/`pdbread` precedent). | `regexp`, file I/O |
| 6.3 | `pcdownsample` / `pcdenoise` | Voxel-grid / random downsample; statistical outlier removal. | grid hash |
| 6.4 | `pcfitplane` / `pcfitsphere` | RANSAC primitive fitting. | RANSAC, `svd` |
| 6.5 | `pcregistericp` | Iterative Closest Point rigid registration of two clouds. | `pdist2`, `svd` (Kabsch) |
| 6.6 | `pcmerge` / `pctransform` | Combine / rigidly transform clouds. | matrix ops |

### 7b. Bag of features + OCR
| # | Surface | Algorithm / notes | Reuses |
|---|---|---|---|
| 6.7 | `bagOfFeatures` | Build a visual vocabulary by `kmeans` over extracted descriptors; `encode` an image to a histogram. | Tier-1, `kmeans` |
| 6.8 | `trainImageCategoryClassifier` / `predict` | Train a multiclass classifier on BoF histograms; classify. | `fitcecoc`/`fitcsvm` |
| 6.9 | `ocr` / `ocrText` | Optical character recognition → recognized text + bounding boxes + confidence. *(stretch — needs a baked character model; a digits/seven-segment subset is the minimal cut)* | template match / DL inference |

**Headline-within-tier**: `cloud = pcread('scene.ply');
cloud = pcdownsample(cloud,'gridAverage',0.1);
plane = pcfitplane(cloud,0.05)` — fit the dominant plane (ground/table).

**Compile/Execute wiring**: `pointCloud` is a classdef; `pcread`/`pcwrite`
are PLY parsers (the Bioinformatics FASTA/`pdbread` precedent);
`pcregistericp`/`pcfitplane` reuse `svd`/`pdist2`; `bagOfFeatures` reuses
the Tier-1 descriptors + Stats `kmeans`; `ocr` is a stretch (baked model).

---

## 8. Phasing & effort summary

The roadmap groups the six tiers into **three shippable phases**:

| Phase | Tiers | Theme | New algorithm | Effort | Closes |
|---|---|---|---|---|---|
| **A — Feature matching + registration** | T1 + T2 | The classical-CV core: detect/describe/match features, estimate the transform, warp | corner detectors, HOG/binary descriptors, `matchFeatures`, RANSAC `estgeotform2d` | **~4 wk** | `feature_match_panorama.m` headline |
| **B — Detection utils + motion** | T3 + T4 | Bounding-box/annotation utilities + optical flow | IoU/NMS, `insert*` raster, Lucas-Kanade/Horn-Schunck flow | **~3 wk** | `optical_flow_motion.m` |
| **C — Geometry + application layer** | T5 + T6 | Camera geometry/stereo + point clouds + bag-of-features + OCR | fundamental/essential, triangulation, disparity, ICP, PLY I/O, BoF | **~5 wk** | `stereo_depth.m`, point-cloud + BoF demos |

**Full toolbox ≈ 12 weeks.** **Phase A alone (~4 wk) is the recommended
first cut** — it is self-contained (rides the shipped Image + linear-algebra
substrate), closes the canonical feature-matching/registration workflow, and
unblocks the rest (Phase B's annotation feeds detection demos; Phase C's
geometry consumes Phase-A correspondences).

**Per-tier dependency notes**:
- T2 depends on T1 (RANSAC estimates over matched points); T2's warp reuses
  the shipped `imwarp`.
- T3 + T4 are independent of T1/T2 (bbox utils + flow need only the image
  substrate) — shippable in parallel.
- T5 depends on T1 (correspondences) + T2 (fundamental matrix); the heaviest
  geometry tier.
- T6 reuses the most (Stats `kmeans`/`fitcecoc` for BoF; Tier-1 descriptors;
  `svd`/`pdist2` for point clouds) and introduces the `pointCloud` classdef.

---

## 9. Carve-outs (explicitly out of scope)

- **Deep-learning detectors / segmenters** (chapters 2–8, 15–16): YOLO
  v2/v3/v4/X, SSD, Faster/Fast/R-CNN, Mask R-CNN, SOLOv2, Grounding DINO,
  semantic/instance/keypoint segmentation, video classification (R(2+1)D /
  SlowFast), anomaly detection (FCDD/PatchCore/EfficientAD), ReID/DeepSORT,
  Vision Transformer, SAM/AnomalyCLIP — all need trained network weights +
  the training loop; they ride the shipped `dlnet` but the demo wiring +
  pretrained-weight import is deferred (consistent with the project's
  repeated DL-training carve-out).
- **Apps**: Camera Calibrator / Stereo Camera Calibrator, Image Labeler /
  Video Labeler / Ground Truth Labeler, the OCR Trainer, Experiment
  Manager, Deep Network Designer — the entire interactive labeling +
  calibration UI surface (chapters 10, 17, 24).
- **Visual SLAM / SfM pipelines** (chapters 1, 18, 19): `monovslam` /
  `stereovslam` / `rgbdvslam`, `imageviewset`/`worldpointset`, bundle
  adjustment, loop closure, NeRF — large stateful pipelines; the *building
  blocks* (feature matching, `estgeotform2d`, `triangulate`,
  `estimateFundamentalMatrix`) are in scope via Tiers 1/2/5.
- **Simulink** block library (chapter 12) and the **OpenCV interface**
  (`mexOpenCV`, OpenCV importer — chapter 21).
- **Camera-calibration estimation** (`estimateCameraParameters` from a
  checkerboard image set — the full bundle-adjusted calibration) — the
  *use* of a `cameraIntrinsics`/`cameraParameters` object (undistort,
  triangulate, project) is in scope (Tier-5); estimating it from images is
  deferred.
- **Lidar deep learning + Lidar SLAM** (chapter 11) — the classical
  `pointCloud` / `pcregistericp` / `pcfitplane` surface is in scope (Tier-6);
  the DL + SLAM pipelines are not.
- **Live video / hardware acquisition** (webcam, `vision.VideoFileReader`
  streaming to scopes, Raspberry Pi deployment), the **Barcode** /
  full **OCR-language-pack** readers.

---

## 10. Compiler traps to watch (from sibling-toolbox experience)

- **`for x = vec` (for-each over a variable vector) does NOT lower under
  ReplMode/-dap** — the Bluetooth lesson; examples that loop over a point
  set / frame list must use the `for i = 1:N` range form + index, or the
  `jit_parity_sweep.py --gate` lane flags them (AOT is unaffected).
- **Four runtime-source lists**: a new `runtime_vision.cpp` must be
  registered in **CMakeLists.txt** (×2: sources + strict-cast), **the
  Run-test harness `test/Run/run_tests.sh`**, AND **the examples sweep
  `test/Examples/run_sweep.sh`** — plus the JIT/DAP gate is a 4th lane (the
  Bioinformatics/Bluetooth lesson).
- **Struct/object matrix fields default to `get_f64`** — a `cornerPoints`
  `.Location` or `pointCloud` `.Location` matrix field read from a
  builtin-returned object must be tagged in `MatStructFields` (the
  Bioinformatics `Payload` / `fastaread` fix); prefer the **function form**
  returning a bare matrix where the object only carries data.
- **Multi-return splitter**: `[features,validPoints]=extractFeatures(...)`,
  `[tform,inliers]=estgeotform2d(...)`, `[disparityMap,...]` — use the
  existing splitter; `numel` of a runtime result is 0 and `~`-ignore-output
  is unsupported (recurring Stats trap).
- **`fprintf` of a comparison / reduction result** doesn't lower — report
  inlier counts via `nnz(inliers)` into a variable then `%.0f`, not
  `fprintf('%d', a==b)`; `%d` of a double prints 0 → use `%.0f`.
- **Deterministic RANSAC / k-means**: seed the RNG (`rng default`) so
  inlier counts + BoF vocabularies are reproducible; pin platform-stable
  thresholds, not exact floats, for the iterative geometry (the RL/Stats
  precedent).
- **CMake build enforces `-Werror=old-style-cast`** (harness doesn't) — use
  `static_cast` throughout `runtime_vision.cpp`; add it to the strict-no-C-
  cast list (the Image/Bioinformatics/Bluetooth precedent).
- **Complex/3-D pitfalls**: RGB images are 3-D arrays — `size(I,3)` and
  `I(:,:,k)` are shipped (Image 3-D-indexing work) but confirm a
  builtin-returned annotated image round-trips as 3-D.

---

## 11. Test & example surface (gating)

- **Gating tests** (`test/Run/vision_*.m`), one per tier headline:
  `vision_features` (T1: `detectHarrisFeatures` count + `matchFeatures` on
  a shifted image recovers the shift, deterministic),
  `vision_geotrans` (T2: `estgeotform2d` on synthetic correspondences with
  a known transform recovers it + inlier count),
  `vision_bbox` (T3: `bboxOverlapRatio` / `selectStrongestBbox` exact IoU +
  NMS), `vision_opticalflow` (T4: `opticalFlowLK` on a translated patch
  recovers the translation), `vision_stereo` (T5: `triangulate` of a known
  3-D point + `estimateFundamentalMatrix` rank/epipolar check),
  `vision_pointcloud` (T6: `pcread`→`pcdownsample`→`pcfitplane` recovers a
  synthetic ground plane).
- **Examples** (`examples/vision/`) mirroring the UG: `feature_match_panorama.m`
  (headline), `optical_flow_motion.m`, `stereo_depth.m`, plus
  `find_object_cluttered.m` (feature-match object localization),
  `hog_digit_features.m` (HOG + `fitcecoc`), `pointcloud_plane_fit.m`.
- **Determinism**: feature counts + match counts are deterministic given a
  fixed image; RANSAC/k-means seeded; `triangulate`/`bboxOverlapRatio` are
  exact; optical flow on a synthetic translation recovers it to a fixed
  tolerance. Any rendering (`imshow`/`insertShape`/quiver) is display-only —
  gating asserts on numeric outputs.
- **Synthetic / bundled fixtures**: generate test images in the `.m`
  (`checkerboard`, shifted/rotated copies, a synthetic stereo pair) so the
  gating lane needs no bundled binaries (the Image `checkerboard` +
  Bioinformatics inline-fixture precedent); a small committed PLY for the
  point-cloud test.

---

## 12. One-line status for MEMORY.md (when shipped)

> Computer Vision Toolbox — roadmap `docs/computer_vision_toolbox_roadmap.md`
> (R2026a UG). High-reuse over the shipped Image Processing + Stats +
> linalg substrate: features over `imgradient`/`imfilter`, `estgeotform2d`
> = RANSAC around shipped `fitgeotform2d`+`imwarp`, geometry/triangulation
> over `svd`, BoF over `kmeans`+`fitcecoc`, point-cloud ICP over
> `svd`/`pdist2`. 6 tiers / 3 phases: A=T1+T2 feature-match+registration
> (~4wk, headline `feature_match_panorama.m`), B=T3+T4 bbox/annotate +
> optical flow (~3wk), C=T5+T6 camera-geometry/stereo + point-cloud/BoF/OCR
> (~5wk). ~12wk full. Objects (`cornerPoints`/`cameraIntrinsics`/
> `pointCloud`/`opticalFlow`) reuse the phytree/DataMatrix/affine2d classdef
> recipe; function forms where JIT state is awkward (Bluetooth precedent).
> Carved: ALL deep-learning detectors/segmenters (YOLO/SSD/R-CNN/Mask/
> SOLOv2/ViT/SAM — need trained weights+training), apps (Camera Calibrator/
> Labelers), Visual SLAM/SfM pipelines/NeRF, Simulink, OpenCV interface,
> camera-calibration-from-checkerboard, Lidar DL/SLAM, live video/hardware.
