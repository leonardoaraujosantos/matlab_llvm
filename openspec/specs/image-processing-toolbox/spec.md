# Image Processing Toolbox Spec

## Purpose
Document the shipped subset of MATLAB's Image Processing Toolbox in `matlab_llvm`: a classical (no-Deep-Learning) image pipeline that reads/writes real image files, filters, transforms, performs morphology, segments, and measures regions. The runtime operates on the `uint8`/`double` pixel-matrix lane plus 3-D RGB arrays.

## Requirements

### Requirement: Image file I/O
The system SHALL read and write real PNG/PGM/PPM/BMP image files to and from pixel matrices.

#### Scenario: Round-trip an image file
- **WHEN** a program calls `imread` on a PNG file and then `imwrite` to a new path
- **THEN** the system SHALL decode the file into a pixel matrix and re-encode a valid image file (matlab_image_imread / matlab_image_imwrite) (doc: docs/image_toolbox_roadmap.md) (src: runtime/toolbox/images/runtime_images.cpp)

### Requirement: Linear filtering and transforms
The system SHALL provide linear/convolutional filtering and 2-D transforms over images.

#### Scenario: Filter and transform an image
- **WHEN** a program calls `imfilter`, `fspecial`, `imgaussfilt`, `imboxfilt`, `medfilt2`, `conv2`, `fft2`/`ifft2`, or `dct2`/`idct2`
- **THEN** the system SHALL return the filtered or transformed result matrix (matlab_imfilter, matlab_image_fspecial, matlab_image_imgaussfilt, matlab_conv2, matlab_fft2_c, matlab_image_dct2) (doc: docs/image_toolbox_roadmap.md) (src: runtime/toolbox/images/runtime_images.cpp)

### Requirement: Geometric transformations
The system SHALL resize, rotate, crop, translate, and warp images, including geometric-transform fitting.

#### Scenario: Resize and warp an image
- **WHEN** a program calls `imresize`, `imrotate`, `imcrop`, `imtranslate`, `imwarp`, or `fitgeotrans`-style fitting with nearest/bilinear/bicubic interpolation
- **THEN** the system SHALL return the transformed image (matlab_image_imresize, matlab_image_imrotate, matlab_image_imwarp, matlab_image_fitgeo_init) (doc: docs/image_toolbox_roadmap.md) (src: runtime/toolbox/images/runtime_images.cpp)

### Requirement: Morphology and binary operations
The system SHALL perform morphological operations and binary image cleanup.

#### Scenario: Apply morphological operators
- **WHEN** a program calls `imerode`, `imdilate`, `imopen`, `imclose`, `imtophat`, `imbothat`, `imfill`, `bwareaopen`, `bwlabel`, or `bweuler`
- **THEN** the system SHALL return the morphologically processed image or labeled regions (matlab_image_imerode, matlab_image_imdilate, matlab_image_imopen, matlab_image_imfill, matlab_image_bwlabel) (doc: docs/image_toolbox_roadmap.md) (src: runtime/toolbox/images/runtime_images.cpp)

### Requirement: Segmentation and region analysis
The system SHALL threshold/segment images and measure region properties.

#### Scenario: Segment and measure regions
- **WHEN** a program calls `graythresh`, `imbinarize`, `imsegkmeans`, edge detection (`edge`), Hough transform (`hough`/`houghpeaks`), or `regionprops`-style measurement (area/centroid/boundingbox/perimeter/orientation/eccentricity)
- **THEN** the system SHALL return the segmented mask or per-region measurements (matlab_image_graythresh, matlab_image_imbinarize, matlab_image_imsegkmeans, matlab_image_edge, matlab_image_hough) (doc: docs/image_toolbox_roadmap.md) (src: runtime/toolbox/images/runtime_images.cpp)

### Requirement: Color conversion, quality metrics, and deblurring
The system SHALL convert color spaces, compute image-quality metrics, and deblur images.

#### Scenario: Color, quality, and restoration
- **WHEN** a program calls `rgb2hsv`/`hsv2rgb`, `rgb2lab`/`lab2rgb`, `mat2gray`, `im2double`/`im2uint8`, `immse`, histogram tools (`imhist`/`histeq`/`imhistmatch`/`adapthisteq`), or Wiener deblurring (`deconvwnr`)
- **THEN** the system SHALL return the converted image, the computed metric, or the restored image (matlab_image_hsv2rgb, matlab_image_lab2rgb, matlab_image_immse, matlab_image_histeq, matlab_image_deconvwnr) (doc: docs/image_toolbox_roadmap.md) (src: runtime/toolbox/images/runtime_images.cpp)
