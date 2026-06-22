## ADDED Requirements

### Requirement: 2-D image signal representation

A mflowLink wire SHALL be able to carry a 2-D image as a flattened, row-major vector
signal whose element count equals `rows · cols · channels`, with the 2-D shape carried
in the signal's existing `(OutRows, OutCols)` metadata (and a `channels` count,
defaulting to 1). The element at image position `(r, c, ch)` SHALL live at flat index
`(r · cols + c) · channels + ch`. The existing 1-D contract (a width-`W` signal has
shape `(1, W)`) MUST remain unchanged.

#### Scenario: Image element count matches its shape
- **WHEN** a block declares an output image of `rows × cols` (grayscale, `channels = 1`)
- **THEN** the wire's signal width is `rows · cols` and its shape metadata is `(rows, cols)`

#### Scenario: Row-major element ordering
- **WHEN** a 2×3 image with rows `[1 2 3; 4 5 6]` flows on a wire
- **THEN** the flattened signal is `[1 2 3 4 5 6]` (row-major), so element `(1, 0)` is at flat index 3

#### Scenario: 1-D signals are unaffected
- **WHEN** an existing scalar or width-`W` vector block is simulated
- **THEN** its signal width and shape `(1, W)` are identical to before this change, and width inference behaves as before

### Requirement: Image-shape conformance

The simulator SHALL treat a block's declared `(rows, cols, channels)` as the source of
truth for a 2-D signal and SHALL reject a model in which a block's declared image shape
is inconsistent with its signal element count.

#### Scenario: Conforming shape accepted
- **WHEN** a block declares shape `(rows, cols, channels)` and produces a signal of width `rows · cols · channels`
- **THEN** the model loads and simulates

#### Scenario: Non-conforming shape rejected
- **WHEN** a block declares an image shape whose `rows · cols · channels` does not equal its signal width
- **THEN** lowering reports a sourced error rather than silently truncating or zero-padding

### Requirement: Image source block

The library SHALL provide `signal_image_source`, a source block that emits a constant
grayscale image of a declared `rows × cols` shape from its parameters.

#### Scenario: Constant image emitted with correct shape
- **WHEN** `signal_image_source` is configured with `rows`, `cols`, and pixel `data`
- **THEN** every step it outputs the `rows · cols` row-major image signal with shape `(rows, cols)`

### Requirement: 2-D image filter block

The library SHALL provide `signal_image_filter`, which convolves/correlates a 2-D image
input with a kernel and outputs an image of the same shape. The kernel SHALL be given as
a matrix literal or a named type (e.g. box, Gaussian, Sobel); border handling SHALL
default to zero-padding.

#### Scenario: Box blur preserves a constant image
- **WHEN** a constant image is filtered with a normalized box kernel
- **THEN** the output equals the input in the interior (a normalized averaging kernel has unity DC gain), with shape preserved

#### Scenario: Sobel edge kernel responds to an edge
- **WHEN** an image with a vertical intensity step is filtered with a horizontal Sobel kernel
- **THEN** the output is non-zero at the edge column and ~zero in flat regions

### Requirement: Per-pixel threshold block

The library SHALL provide `signal_threshold`, which binarizes an image per pixel at a
configurable level, preserving the image shape.

#### Scenario: Pixels binarized at the level
- **WHEN** `signal_threshold` with `level = L` is applied to an image
- **THEN** each output pixel is 1 where the input pixel `> L` and 0 otherwise, with shape preserved

### Requirement: 2-D block authoring recipe

The block-authoring recipe SHALL document the 2-D variant: how a block stamps its output
shape, the row-major index contract, and a `SimulateRun` numeric check over a small
image, so future Vision/Image blocks follow a single documented pattern.

#### Scenario: Recipe covers the 2-D path
- **WHEN** a contributor consults `docs/mflowlink_blocks.md` to add an image block
- **THEN** the recipe specifies the shape-stamping step, the row-major flat-index formula, and an example image regression check
