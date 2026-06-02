# examples/plot/

Self-contained MATLAB programs that exercise the **matlab_plot** runtime —
Cairo-backed, headless, cross-platform (macOS / Linux / iOS). Each example
synthesises its data inline and saves the result to `/tmp/plot_*.png`, so
they double as smoke tests and as reading-order tours of the plotting
surface.

Run any one example with:

```sh
runtime/build_and_run.sh examples/plot/<name>.m /tmp/<name>
/tmp/<name>
```

The plot runtime must be enabled at configure time:

```sh
cmake -B build -DMATLAB_LLVM_WITH_PLOT=ON
```

| File | Demonstrates |
|---|---|
| `sine_wave.m` | Single line plot, MATLAB linespec (`'r--'`), title / xlabel / ylabel / grid, save as PNG. The minimum-viable plot. |
| `multi_series.m` | Two curves on the same axes via `hold on`, distinct linespecs, `legend({'sin(x)', 'cos(x)'})`. |
| `bar_chart.m` | Categorical bar chart via `bar(x, y)`. Bars baseline at zero; widths picked from the x-spacing. |
| `scatter_demo.m` | `scatter(x, y)` — point marker rendering for a noisy linear relationship. |
| `imshow_demo.m` | `imshow(M)` — grayscale display of a 2-D Gaussian bump; data autoscaled to [black, white]. |
| `subplot_grid.m` | `subplot(2, 2, i)` — mixed 2×2 grid of line, scatter, bar, and image axes within one figure. |
| `colormap_demo.m` | Same Gaussian bump rendered with `colormap('parula' / 'viridis' / 'jet' / 'hot')` for visual comparison. Recognised: `gray`, `parula`, `jet`, `viridis`, `hot`, `cool`. |
| `imagesc_demo.m` | `imagesc(Z)` heatmap with `colorbar` — pseudocolor display with axis ticks (vs `imshow` which fills the area). |
| `contour_demo.m` | `contour(Z)` of a peaks-like surface — 10 auto-picked levels via marching squares. |
| `axis_demo.m` | `axis equal` — equal scaling on x and y so a unit circle renders round. Compare default vs `equal` side by side. |
| `log_axes.m` | `semilogy` / `semilogx` / `loglog` — exponential decay is straight on `semilogy`, power law is straight on `loglog`. Decade ticks (1, 10, 100…) auto-generated. |
| `errorbar_demo.m` | `errorbar(x, y, e)` — line plot with vertical bars and hat caps. Symmetric and asymmetric variants supported. |
| `stem_stairs.m` | `stem` (DSP convention: stems from y=0 baseline + circle markers) and `stairs` (sample-and-hold polyline). |
| `histogram_area.m` | `histogram(data, nbins)` of synthetic samples + `area(x, y)` filled curve down to baseline. |
| `text_demo.m` | `text(x, y, str)` annotations placed at data coordinates. |
| `plot3_helix.m` | `plot3(x, y, z)` — 3-D line via software projection (orthographic, MATLAB's default `view(-37.5°, 30°)`). |
| `surf_mesh.m` | `mesh(Z)` and `surf(Z)` — wireframe and filled 3-D surface via tessellation + painter's algorithm depth sort, shaded by colormap. |

### Animation & video (`getframe` + `VideoWriter`)

These capture a figure each step and encode a video file. They need video
support at configure time (links libav):

```sh
cmake -B build -DMATLAB_LLVM_WITH_PLOT=ON -DMATLAB_LLVM_WITH_PLOT_FFMPEG=ON
```

Without that flag the scripts still run, but `VideoWriter` reports that video
support is disabled instead of writing a file. See
[`../../docs/plotting.md`](../../docs/plotting.md) §4.

| File | Demonstrates |
|---|---|
| `videowriter_sine.m` | The minimal video loop: `VideoWriter('out.mp4','MPEG-4')` → `v.FrameRate=30` → `open` → per-frame `plot` + `writeVideo(v, getframe(gcf))` → `close`. Fixed `xlim`/`ylim` keep frame size constant. |
| `animation_orbit.m` | Dynamic-property animation pattern: a marker travels a Lissajous curve over a static path, fixed `axis equal` limits, encoded to MP4. |
| `animation_fourbar.m` | A four-bar linkage (crank-rocker) — vector-loop closure solved per frame via circle-circle intersection, four bars + joints redrawn each step, encoded to MP4. |
| `animation_surf_wave.m` | An animated 3-D `surf` (radial ripple) encoded as **Motion JPEG AVI** — exercises the second profile and the `v.Quality` property. |

## Output formats

Every example writes a PNG. Swap the extension to `.svg` or `.pdf` in
`saveas(gcf, ...)` to get vector output — same draw code, three sinks
(Cairo image / SVG / PDF surfaces).

## Notes

- These are demonstration programs, not regression tests. Runtime tests
  for the plot library live under
  [`../../test/Runtime/test_plot_*.cpp`](../../test/Runtime/) and link
  the C ABI directly without going through the JIT.
- The plot codegen lowering is in flight; until it lands, these `.m`
  files document the target API. The runtime layer (`matlab_plot.h` /
  `runtime/plot/`) is fully exercised by the C-ABI tests.
