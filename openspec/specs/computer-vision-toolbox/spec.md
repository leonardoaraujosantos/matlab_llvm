# Computer Vision Toolbox Spec

## Purpose
Document the shipped classical-core subset of MATLAB's Computer Vision Toolbox in `matlab_llvm`: feature detection/extraction/matching, geometric-transform and pose estimation, stereo/optical-flow, and point-cloud processing layered on the shipped image substrate.

## Requirements

### Requirement: Feature detection and extraction
The system SHALL detect and describe image keypoints.

#### Scenario: Detect and describe features
- **WHEN** a program calls FAST/Harris/min-eigen corner detection, or HOG/LBP feature extraction
- **THEN** the system SHALL return detected points or feature descriptors (matlab_vision_fast, matlab_vision_harris, matlab_vision_mineigen, matlab_vision_hog, matlab_vision_lbp, matlab_vision_extract) (doc: docs/computer_vision_toolbox_roadmap.md) (src: runtime/toolbox/vision/runtime_vision.cpp)

### Requirement: Feature matching
The system SHALL match descriptors and suppress overlapping detections.

#### Scenario: Match features and run NMS
- **WHEN** a program calls `matchFeatures`-style matching or non-maximum suppression with IoU
- **THEN** the system SHALL return matched index pairs or the suppressed box set (matlab_vision_match, matlab_vision_nms, matlab_vision_bboxiou, matlab_vision_bbox2pts) (doc: docs/computer_vision_toolbox_roadmap.md) (src: runtime/toolbox/vision/runtime_vision.cpp)

### Requirement: Geometric transform and pose estimation
The system SHALL estimate geometric transforms and multi-view geometry.

#### Scenario: Estimate transform and fundamental matrix
- **WHEN** a program calls `estgeotform`-style transform estimation, fundamental-matrix estimation, or 3-D triangulation
- **THEN** the system SHALL return the estimated transform, matrix, or 3-D points (matlab_vision_estgeotform, matlab_vision_fundmatrix, matlab_vision_triangulate, matlab_vision_reconstruct) (doc: docs/computer_vision_toolbox_roadmap.md) (src: runtime/toolbox/vision/runtime_vision.cpp)

### Requirement: Stereo and optical flow
The system SHALL compute disparity maps and optical flow.

#### Scenario: Compute disparity and flow
- **WHEN** a program calls stereo disparity or Horn-Schunck / Lucas-Kanade optical flow
- **THEN** the system SHALL return the disparity map or flow field (matlab_vision_disparity, matlab_vision_ofhs, matlab_vision_oflk) (doc: docs/computer_vision_toolbox_roadmap.md) (src: runtime/toolbox/vision/runtime_vision.cpp)

### Requirement: Point cloud processing
The system SHALL read/write, downsample, fit, and register point clouds.

#### Scenario: Process a point cloud
- **WHEN** a program calls point-cloud read/write, downsampling, plane fitting, or ICP registration
- **THEN** the system SHALL return the processed cloud or registration transform (matlab_vision_pcread, matlab_vision_pcwrite, matlab_vision_pcdownsample, matlab_vision_pcfitplane, matlab_vision_pcicp) (doc: docs/computer_vision_toolbox_roadmap.md) (src: runtime/toolbox/vision/runtime_vision.cpp)

### Requirement: Annotation overlays
The system SHALL draw markers and shapes onto images.

#### Scenario: Annotate an image
- **WHEN** a program calls `insertMarker` or `insertShape`
- **THEN** the system SHALL return the annotated image (matlab_vision_insertmarker, matlab_vision_insertshape) (doc: docs/computer_vision_toolbox_roadmap.md) (src: runtime/toolbox/vision/runtime_vision.cpp)
