# Plotting — Feature Status & Roadmap

Reference for the **matlab_plot** runtime in `runtime/plot/` and its
matlabc codegen wiring (`lib/MLIR/Passes/LowerPlot.cpp`). Covers what's
shipped and what's still open. Companion to
[`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md) and
[`feature_status.md`](feature_status.md).

The runtime is a pure-C++/Cairo backend that compiles to PNG/SVG/PDF in
memory or on disk. Headless: no subprocess, no display server, no temp
files. Targets macOS, Linux, and iOS from the same codebase.

Enable with:

```sh
cmake -B build -DMATLAB_LLVM_WITH_PLOT=ON
```

Cairo must be discoverable via `pkg-config` (`cairo`, `cairo-svg`,
`cairo-pdf`). Homebrew supplies these on macOS; `libcairo2-dev` on
Debian/Ubuntu.

Video export (`getframe` + `VideoWriter`, §4 Tier A/B) is **on by
default** within a `WITH_PLOT` build:

```sh
cmake -B build -DMATLAB_LLVM_WITH_PLOT=ON            # video included
```

This links libav directly, so a `WITH_PLOT` build needs the FFmpeg dev
libraries (`brew install ffmpeg` on macOS; `apt install libavcodec-dev
libavformat-dev libavutil-dev libswscale-dev` on Debian/Ubuntu) — the
same hard-dependency posture as Cairo. On a host without them, opt out
with `-DMATLAB_LLVM_WITH_PLOT_FFMPEG=OFF`: plot scripts still compile and
run, and `VideoWriter` reports that video support is disabled.

## 0. Reading guide

- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started.
- **Layer**: which part of the stack the feature touches.
  - **Runtime** — `runtime/plot/{c_api,figure,cairo_render,colormap,contour}.cpp`
  - **Sema** — `lib/Sema/Resolver.cpp` builtin registry
  - **Codegen** — `lib/MLIR/Passes/LowerPlot.cpp` `matlab.call_builtin` rewriter
- **Effort tags** roughly mirror the rest of the project: a *session* is
  half a day; a *week* is ≈5 sessions.

## 1. Architecture

```
   .m source ──▶  matlabc Lex/Parse/Sema  ──▶  MLIR (matlab.call_builtin)
                        │
                        │ Resolver registers plot/figure/title/...
                        │ as Builtin so name resolution succeeds
                        ▼
                  LowerPlot pass  ──▶  llvm.call @matlab_plot2(ptr, ptr)
                  (rewrites call_builtin    +  @matlab_title(ptr, i64)
                   into runtime calls)      +  ... etc.
                        │
                        ▼
                  matlabc JIT
                  (DynamicLibrarySearchGenerator finds matlab_*
                   symbols linked into the matlabc binary)
                        │
                        ▼
                  runtime/plot — Figure/Axes state machine,
                                  Cairo painter, colormap LUTs, ...
                        │
                        ▼
                  PNG / SVG / PDF bytes  (on disk or in memory)
```

The plot runtime is linked into `matlabc` so the JIT can resolve the
~70 `matlab_*` plot symbols against the running process — same lookup
mechanism as the rest of `matlab_runtime.cpp`. `runtime-plot-hello` and
`runtime-plot-basic` exercise the same C ABI from C++ tests, no JIT
involved, so the runtime can be regression-tested in isolation.

## 2. Already shipped

### 2.1 Plot kinds — 2D

| Function | Status | Layer | Notes |
|---|---|---|---|
| `plot(y)`, `plot(x, y)` | ✅ | all | `matlab_plot1` / `matlab_plot2` |
| `plot(x, y, 'fmt')` | ✅ | all | Linespec parser handles colour letters (rgbcmykw), line styles `-`/`--`/`:`/`-.`, markers `o + * x s d ^`. `matlab_plot_fmt`. |
| `scatter(x, y)` | ✅ | all | Filled circle markers. |
| `bar(y)`, `bar(x, y)` | ✅ | all | Auto bar-width = 70 % of smallest x-gap. |
| `stem(y)`, `stem(x, y)` | ✅ | all | DSP convention: vertical line from y=0 + filled marker at top. |
| `stairs(y)`, `stairs(x, y)` | ✅ | all | Sample-and-hold polyline. |
| `area(y)`, `area(x, y)` | ✅ | all | Filled region between curve and y=0 baseline (35 % alpha) plus solid outline. |
| `errorbar(x, y, e)` | ✅ | all | Symmetric vertical bars + 4 px hat caps + centre dot. |
| `errorbar(x, y, neg, pos)` | ✅ | all | Asymmetric. |
| `histogram(data)` / `histogram(data, n)` | ✅ | all | Bins via flush bars (`bar_half_factor=0.49`). `n<=0` picks `round(sqrt(numel))`. |
| `imshow(M)` | ✅ | all | Fills plot area, no ticks; row 1 at top (`y_inverted`). |
| `imagesc(M)` | ✅ | all | Heatmap with axis ticks; honours active colormap. |
| `pcolor(M)` | ✅ | all | Alias of imagesc for now (cell-vertex semantics not yet distinguished). |
| `contour(Z)` / `contour(Z, levels)` | ✅ | all | Marching squares. Auto picks 10 levels evenly spaced strictly between min/max. |
| `contourf(Z)` | ✅ | all | Pragmatic implementation: `imagesc` + `contour` overlay. Visually matches MATLAB's filled contour for smooth fields. |
| `quiver(x, y, u, v)` | ✅ | all | Auto-scaled arrows with 6 px arrowheads at 25 °. |

### 2.2 Plot kinds — 3D

| Function | Status | Layer | Notes |
|---|---|---|---|
| `plot3(x, y, z)` | ✅ | all | Orthographic projection from `view(az, el)`. |
| `plot3(x, y, z, 'fmt')` | ✅ | all | Linespec applied to 3D line. |
| `mesh(Z)` / `mesh(X, Y, Z)` | ✅ | all | Wireframe via tessellation + painter's-algorithm depth sort. |
| `surf(Z)` / `surf(X, Y, Z)` | ✅ | all | Filled surface, per-face Lambertian-style shading via active colormap (z-centroid → colour). Thin black edges for definition. |

### 2.3 Decoration

| Function | Status | Notes |
|---|---|---|
| `title('s')`, `xlabel('s')`, `ylabel('s')`, `zlabel('s')` | ✅ | sans-serif, title bold 14 pt, labels 12 pt. ylabel rotated 90 °. zlabel only for 3D axes. |
| `text(x, y, 's')` | ✅ | At data coordinates, clipped to plot area. Cleared on non-hold plot. |
| `legend('a', 'b', ...)` (varargs) | ✅ | Upper-right inset box; per-series swatch matches kind (line/marker/dot/rect). |
| `legend({'a', 'b'})` (cell) | 🟡 | Runtime side ready (walks cell via `matlab_cell_get_mat` and decodes char-mats). Codegen dispatches the single-PtrTy case. Blocked on `LowerTensorOps` plumbing for `matlab_cell_set_mat`'s `tensor<…xi8>` operands. |
| `colorbar` | ✅ | Vertical strip on the right with 5 evenly-spaced tick labels reading the image data range. |
| `grid on` / `grid off` | ✅ | Light-gray background grid at tick positions. |
| `hold on` / `hold off` | ✅ | Controls accumulate-vs-replace on next plot call. |

### 2.4 Layout

| Function | Status | Notes |
|---|---|---|
| `figure` | ✅ | Returns opaque `matlab_figure*`. |
| `figure(n)` | ✅ | Switches to / creates figure with id `n`. |
| `gcf` | ✅ | Current figure (per-thread). |
| `close` (no arg) / `close('all')` | ✅ | Both forms call `matlab_close_all`. |
| `subplot(m, n, i)` | ✅ | Reshapes the axes grid to m × n; new cells start empty. Each cell paints independently with its own Layout margins. |

### 2.5 Style

| Function | Status | Notes |
|---|---|---|
| Auto colour cycling | ✅ | MATLAB R2014b+ palette: blue, orange, yellow, purple, green, light blue, dark red. Resets on figure / non-hold replacement. |
| `'LineWidth', w` | ✅ | Code-gen strips trailing Name/Value pairs; runtime via `matlab_set_series_linewidth`. |
| `'Color', [r g b]` | ✅ | RGB triplet path uses `matlab_set_series_color_mat`. |
| `'Color', 'r'` | ✅ | Single-letter colour decoded at IR-emit time into RGB constants. |
| `'LineStyle', '-' / '--' / ':' / '-.'` | ✅ | All four cairo dash patterns wired. |
| `'Marker', 'o' / '+' / '*' / 'x' / 's' / 'd' / '^'` | ✅ | `'.'` mapped to a small `'o'`. `'none'` clears the marker. |
| `'MarkerSize', s` | ✅ | Default 4 px. |
| `'DisplayName', 's'` | ✅ | Stored on the series; future legend-without-args path will read it. |
| `'MarkerFaceColor'`, `'MarkerEdgeColor'` | 🟡 | Codegen routes through `set_series_color`; runtime treats both as foreground colour for now (no separate face/edge state). |

### 2.6 Axes

| Function | Status | Notes |
|---|---|---|
| `xlim([lo hi])`, `ylim([lo hi])` | ✅ | Numeric form. |
| `axis('opt')` | ✅ | `equal` / `square` / `tight` / `off` / `on`. |
| `axis([xmin xmax ymin ymax])` | ✅ | Numeric 4-element form via `matlab_axis_lims`. |
| `box on` / `box off` | ✅ | Frame on/off independent of ticks. |
| `view(az, el)` | ✅ | 3D camera angles in degrees. Default (-37.5°, 30°). |
| `xline(v)`, `yline(v)`, `xline(v, 'lbl')`, `yline(v, 'lbl')` | ✅ | Reference lines spanning the plot area, optional label. |
| `xticks(v)`, `yticks(v)` | ✅ | Override auto tick locations with a row vector. |
| `xticklabels('a', 'b', ...)`, `yticklabels(...)` | ✅ | Custom labels at the active tick positions. |
| `yyaxis('left')` / `yyaxis('right')` | ✅ | Independent left/right y-axes; non-hold plot replaces only the active side. |

### 2.7 Scales

| Function | Status | Notes |
|---|---|---|
| `loglog`, `semilogx`, `semilogy` | ✅ | Decade ticks (1, 10, 100, ...) with magnitude-aware label format (`1e-3` for very small/large). Non-positive data clamped at compute-data-range time. |

### 2.8 Colormaps

| Map | Status | Implementation |
|---|---|---|
| `gray` | ✅ | Linear black→white. |
| `parula` | ✅ | 16-sample LUT (MATLAB R2014b+ default). |
| `jet` | ✅ | Computed analytically. |
| `viridis` | ✅ | 16-sample LUT. |
| `hot` | ✅ | Computed analytically. |
| `cool` | ✅ | Computed analytically. |

`colormap('name')` switches the active map for the current axes.
Unknown names fall back to `parula`.

### 2.9 Output

| Function | Status | Notes |
|---|---|---|
| `saveas(h, 'path.png')` | ✅ | Figure handle ignored (we use the current figure); extension dispatch. |
| `saveas(gcf, 'path')` | ✅ | The orphan `matlab.make_handle` from `gcf`-as-handle is swept by `LowerPlot`. |
| `print('path')` | ✅ | Same as saveas with one arg. |
| C-ABI in-memory: `matlab_render_png/svg/pdf` | ✅ | Returns malloc'd buffer; caller releases with `matlab_plot_buffer_free`. |
| `getframe` + `VideoWriter` → `.mp4` / `.avi` | ✅ (behind `…_FFMPEG`) | Animation capture + video export. See §4 Tier A/B. |
| Extensions | `.png`, `.svg`, `.pdf` ✅ | Inferred from path. Unknown extension returns `-1`. |

### 2.10 matlabc integration

| | Status |
|---|---|
| Build: matlabc with `WITH_PLOT=ON` and LLVM/MLIR | ✅ |
| ~70 plot symbols exported (visible to JIT `DynamicLibrarySearchGenerator`) | ✅ |
| Sema: 47 plot names registered as builtins | ✅ |
| MLIR Lowering: `figure`/`gcf` typed as `PtrRet`; rest default `none` (void) | ✅ |
| `LowerPlot.cpp`: rewrites `matlab.call_builtin` → `llvm.call @matlab_*` | ✅ |
| Pass wired into all three matlabc pipelines (default, REPL, DAP) | ✅ |
| Property/value pair stripping + setter follow-up calls | ✅ |
| Dual string-op handling (`const_char` + `const_str`) for command syntax (`grid on`) | ✅ |
| Dead-op sweep for orphaned `const_str` / `const_char` / `make_handle` | ✅ |

## 3. Roadmap

### Tier 1 — small remaining gaps from the shipped surface

| Item | Effort | Notes |
|---|---|---|
| `legend({'a','b'})` cell-of-strings end-to-end | 1 session | Runtime + dispatch ready; needs `LowerTensorOps` to convert `matlab.const_char` (tensor<1xNxi8>) to `matlab_mat *` when feeding `matlab_cell_set_mat`. |
| `MarkerFaceColor` / `MarkerEdgeColor` distinct from line colour | 1 session | Add separate `face_r/g/b/a`, `edge_r/g/b/a` to `Series`; marker painter switches by glyph filled vs outlined. |
| `LineWidth` → 3D series | already works |  |
| `legend('Location', 'NorthWest')` | 1 session | Add `Axes::legend_location` enum; `draw_legend` consults it for box anchor. |
| `print -r300 path` (DPI) | 1 session | New runtime entry that takes DPI; cairo image surface created with scaled width/height. |

### Tier 2 — common in scientific plots

| Item | Effort | Notes |
|---|---|---|
| `polarplot(theta, r)` | 1 session | New `SeriesKind::Polar`. Renderer paints a polar grid (concentric circles + radial spokes) and projects (θ, r) to (x, y). |
| Polar coordinate axes (general) | 1 week | Architecturally bigger — `Axes::projection` enum (Cartesian, Polar, Geographic). Tick generation, gridlines, labels all need polar paths. |
| `tiledlayout(m, n)` / `nexttile` | 3 sessions | Modern subplot replacement. Internally maps to `subplot` for now; could give tighter spacing in v2. |
| TeX/LaTeX in labels (`'Interpreter', 'latex'`) | 1 week | Either embed `lualatex` shell-out (kills iOS) or implement a TeX subset (subscripts, superscripts, common Greek letters) directly in the label painter. The subset path is what we'd ship. |
| `surf(X, Y, Z, C)` with explicit colour matrix | 1 session | Add `Series::face_colors` (per-vertex). Painter samples C instead of z when present. |
| `shading flat / interp / faceted` | 2 sessions | Currently fixed at "faceted" (per-face flat + thin black edges). `flat` would drop edges; `interp` needs Gouraud-style cairo gradients per quad. |
| `bar3(Z)` | 2 sessions | 3D bars via cuboid tessellation + painter's-algorithm. |
| `scatter3(x, y, z)` | 1 session | Project, sort by depth, draw filled markers. |
| `fill(x, y)` and `fill3(x, y, z)` | 1 session | Filled polygons in 2D / 3D. |
| `pie(x)`, `pie3(x)` | 1 session | Sector painter + label per slice. |

### Tier 3 — less common

| Item | Effort | Notes |
|---|---|---|
| `boxplot(data)` | 2 sessions | Quartile + whisker painter; per-group support adds another session. |
| `violinplot` (R2025a+) | 3 sessions | KDE estimation + symmetric area painter. |
| `heatmap` (cell-shaded with axis labels) | 2 sessions | Different from `imagesc`: discrete categories on both axes, cell text overlays. |
| `geoplot`, `geobubble` | weeks | Map projection + tile fetch — out of scope for the headless runtime story. |
| 3D filled: `isosurface`, `slice`, `streamline`, `streamtube`, `coneplot`, `streamslice` | 1–2 weeks | Volume/vector-field family. Requires marching cubes for isosurfaces. |
| `quiver3(x, y, z, u, v, w)` | 1 session | Same painter as 2D quiver but project endpoints first. |
| `waterfall(X, Y, Z)`, `ribbon(Y)` | 2 sessions | Both are surface-variant painters; `ribbon` builds 3D strips from 2D series. |
| `meshc` / `surfc` (mesh / surf with contour at base) | 1 session | Compose existing mesh/surf with a contour drawn at z = z_lo. **[2026-05 in-flight slice]** |
| `peaks(N)` demo data generator | 1 session | Closed-form 3-return `[X, Y, Z] = peaks(N)`. The canonical "draw something interesting" 3-D fixture used in every MATLAB tutorial. Pairs with `surfc` above. **[2026-05 in-flight slice]** |
| Additional colormaps (`magma` / `inferno` / `plasma` / `cividis` / `turbo`) | 1 session | Matplotlib-parity lookup tables. Lets users porting matplotlib scripts find the colormap they expect; closes the surprise gap users hit today (we ship gray / parula / jet / viridis / hot / cool only). **[2026-05 in-flight slice]** |
| `triplot` / `trisurf` / `trimesh` (triangulated meshes) | 1 week | Read triangle index list; painter dispatches to existing mesh/surf primitives per face. |
| Lighting: `light`, `lighting flat/gouraud/phong`, `material shiny/dull/metal`, `camlight` | 1 week | Per-face Lambertian (flat) is essentially what `surf` does today; `gouraud`/`phong` need per-vertex normals + cairo gradient tricks. |
| Multi-series syntax `plot(x,y1,'r-', x,y2,'b--')` | 1 session | Detect alternating (vec, vec, opt-fmt) groups in `LowerPlot`. |
| `image()` raw RGB, `montage`, `imshowpair` | 1 session each | Variants of the existing image painter. |
| Animation: `drawnow`, `animatedline`, `comet`, `VideoWriter` | partial | `getframe` + `VideoWriter` (MP4/AVI) shipped — see §4 Tier A/B. `drawnow` real semantics / `animatedline` / `comet` still open. |

### Tier 4 — function-expression plots (need symbolic backend)

| Item | Effort | Notes |
|---|---|---|
| `fplot('sin(x)', [0 10])` | 1 week | Lazy MATLAB-expression evaluation over an x grid. Could plug into `MATLAB_LLVM_WITH_SYM` SymPP path (already in repo) or call back into the JIT to eval an anonymous function. |
| `fmesh`, `fsurf`, `fcontour`, `fimplicit` | 2 weeks | Multivariate variants of fplot. |
| `ezplot` family | 1 session | Older alias of `fplot`; a wrapper. |

### Tier 5 — won't make sense headless

These are explicitly out of scope for the matlab_plot runtime. Tracked
here so it's clear they're not just "missing".

| Feature | Reason |
|---|---|
| Pan, zoom, rotate3d, datacursormode | Require a live window with mouse input. |
| `ginput()` | Cursor picking. |
| App Designer / `uifigure` / `uigauge` | UI authoring system. |
| Live Editor inline plots | IDE-specific. |
| Plot tools (interactive editor) | UI workflow. |

### Tier 6 — output enhancements

| Item | Effort | Notes |
|---|---|---|
| Background transparency | 1 session | `cairo_format_argb32` with `cairo_paint_with_alpha` instead of opaque white background; CLI flag to opt in. |
| EPS, TIF, BMP, JPEG | 1 session each | Cairo doesn't support these natively; we'd render to PNG bytes and pipe through `libtiff` / `libjpeg` / etc., or rely on caller post-processing. |
| Animated GIF | 2 sessions | Render frame-by-frame, write via `libgif`. |
| Multi-page PDF | 1 session | `cairo_show_page` between renders; expose via a multi-figure save API. |

## 4. Animation & interactivity roadmap

Animation slots cleanly into the one-shot Cairo render path: each frame
is a fresh paint, captured via the existing in-memory PNG path
(`matlab_render_png`) and either streamed to the IDE through the
sentinel channel (§6) or written to a video container.

Interactivity is split into two tracks:
- **Tier D — HTML export.** Headless-friendly. A self-contained SVG/HTML
  bundle with vanilla-JS pan/zoom (and three.js for 3D). Works from
  REPL, DAP, batch, and iOS.
- **Tier E — Native window.** Opt-in CMake flag, desktop only. SDL2 +
  cairo on a window surface. Implements true `pan` / `zoom` /
  `rotate3d` / `datacursormode` / `ginput` / callbacks; the default
  headless build is untouched.

Animation tiers stay fully headless. The IDE / REPL already streams
figures via the sentinel channel after every statement (post-input
flush in `tools/matlabc/main.cpp`), so animation only needs to extend
the existing pipe. This section expands on the Animated-GIF / Multi-page
PDF teasers in §3 Tier 6.

### Tier A — Animation core (output-only, fully headless)

| Item | Effort | Notes |
|---|---|---|
| `drawnow` real semantics | 1 session | Today `matlab_drawnow` is a thin shim calling `matlab_ide_emit_all_figures`. Add `Figure::dirty` flag; flush only dirty figures; `drawnow('limitrate')` rate-limits to 20 Hz; `drawnow('expose')` becomes a no-op alias. |
| `animatedline()` + `addpoints` / `clearpoints` | 2 sessions | New `SeriesKind::AnimatedLine` with growable x/y/z buffers. Returns a handle (numeric token, like figure ids). Name/Value pairs (`'MaximumNumPoints'`, `'LineStyle'`, …) route through the existing prop-setter path in `LowerPlot.cpp`. |
| `comet(x, y)` / `comet3(x, y, z)` | 1 session | Sugar over animatedline: per-step `addpoints` + `drawnow` + sleep. MATLAB's head/body/tail = three series with different alphas. |
| `getframe()` / `getframe(h)` | ✅ | Shipped. `matlab_getframe` renders the current figure to a raw ARGB32 raster (`cairo_render::render_raw`, not a re-decoded PNG — encoders want pixels) and returns an opaque `matlab_frame *`. Frames live in a per-thread registry freed on `close_all` / thread exit, so scripts never free them. A handle argument (`getframe(gcf)`) is accepted and ignored. |
| `movie(F, n, fps)` | 1 session | In REPL/IDE: streams frames via the sentinel channel at `fps`. In batch: optional multi-page PDF, otherwise no-op. |
| `pause(t)` integration | trivial | Already exists in `matlab_runtime`; animation loops just need to yield to it. |

### Tier B — Animation containers (write video files)

| Item | Effort | Notes |
|---|---|---|
| Multi-page PDF via `drawnow` | 1 session | `cairo_show_page()` between paints when output is `.pdf`. |
| Animated GIF via `libgif` (giflib) | 2 sessions | Optional dep behind `MATLAB_LLVM_WITH_PLOT_GIF`. `saveas(gcf, 'anim.gif')` after a movie buffer is filled; or `VideoWriter` profile `'GIF'`. Per-frame 256-color quantization at encode time. |
| `VideoWriter` — MP4/H.264 | ✅ (v1) | Shipped behind `MATLAB_LLVM_WITH_PLOT_FFMPEG` (on by default within a `WITH_PLOT` build; links libav directly). API: `v = VideoWriter('out.mp4', 'MPEG-4'); v.FrameRate = 30; open(v); writeVideo(v, getframe(gcf)); close(v);`. `runtime/plot/videowriter.cpp` holds the opaque handle + `matlab_videowriter_new/_new_profile/_set_framerate/_set_quality/_open/_write/_close`; in-process ARGB→YUV420P via swscale, H.264 (libx264) or MJPEG encode. Profiles in v1: `'MPEG-4'` (→ H.264/MP4) and `'Motion JPEG AVI'` (→ MJPEG/AVI); the container is also inferred from the path extension. **v1 limits:** `FrameRate`/`Quality` are set via scalar property assignment (`v.FrameRate = N`); `close(handle)` is assumed to be a VideoWriter (figure-handle close isn't distinguished yet); `'Uncompressed AVI'` / `'Archival'` profiles and `VideoReader` are follow-ups. The symbols always exist; without the flag, VideoWriter reports that video support is disabled instead of writing a bogus file. |
| `VideoReader` | 1 session | Symmetric — libav demux + PNG-decoded frames. Lower priority. |

FFmpeg is on by default within a `WITH_PLOT` build; GIF stays optional.
When video is opted out (`-DMATLAB_LLVM_WITH_PLOT_FFMPEG=OFF`), animation
still produces multi-page PDF + IDE streaming, which covers most REPL
workflows.

### Tier C — REPL / Debug / IDE wiring (the "live plotting" channel)

The REPL already calls `matlab_ide_emit_all_figures` after every
statement (post-input flush in `tools/matlabc/main.cpp`) and the
sentinel format `___MF_FIG_BEGIN___ id=… w=… h=…` is already plumbed
through stdout. Animation only needs minor extensions.

| Item | Effort | Notes |
|---|---|---|
| Sentinel `kind=frame\|figure` header | trivial | IDE distinguishes in-progress animation frames from final figures (replace-in-place vs. accumulate). |
| `drawnow` rate-limit aware of IDE backpressure | 1 session | When IDE is consuming, throttle to `MATLAB_LLVM_IDE_FPS` (default 20). |
| DAP custom event for frames | 2 sessions | DAP today only responds to client requests; add a custom `output` category `"figure"` carrying base64 PNG (or a non-standard `figure` event). Lets you see animations while paused at a breakpoint. |
| REPL figure echoing during loops | 1 session | Inside a `for` loop, no per-iteration `runReplInput` boundary fires — `matlab_drawnow` itself must write to the same fd as REPL stdout. |

### Tier D — Interactive plots via HTML export (still headless)

Pragmatic answer to "interactive plots" without breaking the headless
story: emit a self-contained HTML/SVG bundle.

| Item | Effort | Notes |
|---|---|---|
| `saveas(gcf, 'fig.html')` for 2D | 1 week | Re-use the existing SVG painter, wrap in an HTML shell with ~200 lines of vanilla JS: pan (mousedrag), zoom (wheel), tooltip on nearest point. No external deps; one file you can email. |
| HTML export for 3D | 1 week | Emit a three.js (or twgl) micro-bundle plus surface/mesh/line data as JSON. Rotate / pan / zoom via OrbitControls. |
| Animated HTML (timeline scrubber) | 3 sessions | When a movie buffer is captured, embed frames as `<img>` slideshow + scrubber `<input>`. |

This gives interactive plots that work from REPL / DAP / batch / iOS —
the user just opens the file in any browser.

### Tier E — Interactive plots via optional native window (opt-in, desktop only)

For the closest-to-MATLAB experience (real `pan`, `zoom`, `rotate3d`,
`datacursormode`, `ginput`, callbacks), add a separate window backend
behind a CMake flag. **Opt-in**, **macOS + Linux only**, and the default
headless build remains untouched.

| Item | Effort | Notes |
|---|---|---|
| `MATLAB_LLVM_WITH_PLOT_WINDOW` (SDL2 + cairo image-on-texture) | 1 week | `figure()` opens an SDL window; cairo paints into a CPU surface uploaded as a GL/Metal texture each frame. |
| `pan` / `zoom` / `rotate3d` as mode toggles | 1 week | Mouse-drag mutates `Axes::xlim/ylim` (pan), scales them (zoom), or rotates `Axes::view_az/el` (rotate3d). Each mutation marks the figure dirty → repaint loop. |
| `ginput(n)` blocking | 2 sessions | Drains SDL mouse events on the JIT thread; returns `[x,y]` in axes data coords. Errors in headless build. |
| `datacursormode on` + nearest-point lookup | 2 sessions | k-d-style nearest-neighbor on visible series; tooltip painter. |
| Callbacks (`'ButtonDownFcn'`, `WindowKeyPressFcn`, …) | 1 week | Requires re-entering the JIT from the event thread to invoke a MATLAB anonymous function. The `matlab_dbg_hook` re-entry pattern in `tools/matlabc/main.cpp` is the precedent. |
| `MATLAB_LLVM_WITH_PLOT_WEBSOCKET` (web alternative) | 1 week | Same surface as above but events come over WS from a browser viewing the HTML export. iOS-friendly variant of E. |

### Tier F — Touch points

Discrete files animation/interactivity must extend:

- `runtime/matlab_plot.h` — C ABI. **Done:** `matlab_getframe`, `matlab_videowriter_*`. **Remaining:** `matlab_drawnow_flushed`, `matlab_animatedline_new`, `matlab_addpoints`, `matlab_clearpoints`, `matlab_movie`.
- `runtime/plot/frame.h` — **Done:** internal `matlab_frame` definition (ARGB32 raster) shared by the capture path and the encoder.
- `runtime/plot/videowriter.cpp` — **Done:** the libav encoder + opaque `matlab_videowriter` handle (guarded by `MATLAB_LLVM_WITH_PLOT_FFMPEG`).
- `runtime/plot/cairo_render.{h,cpp}` — **Done:** `render_raw()` (raw frame capture). **Remaining:** multi-page PDF support; frame-rate aware flush.
- `runtime/plot/cairo_dl.cpp` — **Done:** added `cairo_image_surface_get_data` / `_get_stride` / `cairo_surface_flush` for the raw-pixel readback.
- `runtime/plot/figure.h` — `SeriesKind::AnimatedLine`, `Figure::dirty`, `Axes::interaction_mode` enum (remaining). The per-thread frame registry lives in `c_api.cpp`.
- `lib/MLIR/Passes/LowerPlot.cpp` — **Done:** `getframe` / `VideoWriter` / `open` / `writeVideo` in `plotBuiltins()` + `rewriteCallee()`; `close(ptr)` → `matlab_videowriter_close`.
- `lib/MLIR/Lowering.cpp` — **Done:** `getframe` / `VideoWriter` in the `PtrRet` set; `VideoWriterBindings` tag so `v.FrameRate = N` / `v.Quality = N` route to the setters.
- `lib/MLIR/Passes/LowerTensorOps.cpp` — **Done:** lowers the `matlab_videowriter_set_framerate/_set_quality` calls (with scalar→f64 coercion).
- `lib/Sema/Resolver.cpp` — **Done:** `getframe`, `VideoWriter`, `open`, `writeVideo` registered.
- `tools/matlabc/main.cpp` — sentinel `kind=` extension; optional DAP figure event; optional SDL window event loop (remaining).
- `CMakeLists.txt` — **Done:** `MATLAB_LLVM_WITH_PLOT_FFMPEG` (on by default; `WITH_PLOT` umbrella). **Remaining:** `…_GIF`, `…_WINDOW`, `…_WEBSOCKET`.
- `test/Runtime/test_plot_video.cpp` — **Done:** getframe + VideoWriter direct-ABI test (encode assertions gated on the flag, disabled-path asserted otherwise).
- `examples/plot/{videowriter_sine,animation_orbit,animation_fourbar,animation_surf_wave}.m` — **Done** (MP4 + Motion JPEG AVI; 2-D motion, four-bar mechanism, animated 3-D surf). `examples/plot/animatedline_*.m`, `examples/plot/comet_*.m` — remaining.

### Tier G — Explicitly out of scope

- **No emit-c / emit-cpp / emit-python / emit-typescript / emit-systemverilog plumbing.** `LowerPlot` runs only on the JIT/runtime path; emit backends never see plot ops. SystemVerilog won't paint Cairo; Python/TS users have matplotlib/Plotly natively. C/C++ emit could in principle link against `libmatlab_plot`, but not pursued.
- Plot tools / Live Editor / App Designer / `uifigure` — same rationale as the §3 Tier 5 carve-out.
- WebGL-based realtime 3D inside the cairo runtime — too much surface area for the headless story.

### Suggested execution order

1. **A1–A4** (drawnow + animatedline + getframe) — covers >70 % of MATLAB animation scripts in the wild, all headless. ~1 week.
2. **C1–C3** (REPL / DAP wiring) — makes animations *visible* in the IDE during interactive use. ~3 sessions.
3. **B2** (libgif) — first persistent video output. ~2 sessions.
4. **D1** (HTML export, 2D) — first form of "interactive plots". ~1 week.
5. Then choose between **B3** (MP4) / **D2** (3D HTML) / **E** (native window) based on demand.

## 5. Internals reference

### File layout

```
runtime/matlab_plot.h      Public C ABI (declared extern "C")
runtime/plot/c_api.cpp     C ABI shim → calls into the C++ figure/cairo state
runtime/plot/figure.{h,cpp} Figure / Axes / Series / RefLine / TextAnnotation
                            data model + per-thread figure registry
runtime/plot/cairo_render.{h,cpp}  Painters: line/scatter/bar/stem/stairs/area
                                    /errorbar/imshow/imagesc/contour/contourf
                                    /quiver/surf/mesh/plot3 + axes/legend/colorbar
runtime/plot/colormap.{h,cpp}  6 colormap LUTs (gray/parula/jet/viridis/hot/cool)
runtime/plot/contour.{h,cpp}   Marching squares (used by contour and contourf)
lib/MLIR/Passes/LowerPlot.cpp  Codegen pass: matlab.call_builtin → llvm.call
lib/Sema/Resolver.cpp          Builtin name registry (47 plot names)
include/matlab/MLIR/Passes/Passes.h  runLowerPlot declaration
test/Runtime/test_plot_*.cpp   Direct C-ABI tests (no JIT)
examples/plot/*.m              MATLAB example scripts
```

### Codegen flow for a single MATLAB call

`plot(x, y, 'LineWidth', 2);` lowers as follows:

1. **Lex/Parse** → AST `CallOrIndex { name="plot", args=[x, y, "LineWidth", 2] }`.
2. **Sema/Resolver** → name `plot` resolves to `BindingKind::Builtin`.
3. **MIR/Lowering** (`lib/MLIR/Lowering.cpp`) → emits `matlab.call_builtin "plot"` with 4 operands; result type `none` (because `plot` isn't in the F64Ret/PtrRet sets).
4. **LowerPlot pass** (`lib/MLIR/Passes/LowerPlot.cpp`):
   - `countTrailingPropPairs(Op)` finds 1 pair (`"LineWidth", 2`) by walking from the end and matching against `isKnownSeriesProp`.
   - Strips the pair: op now has 2 operands `(x, y)` only.
   - Calls `rewriteCallee`, which recognises 2-operand `plot` and emits `llvm.call @matlab_plot2(x, y)`.
   - Restores the original operands and calls `emitPropSetters`, which emits `llvm.call @matlab_set_series_linewidth(2.0)` after the main call.
   - Erases the original `matlab.call_builtin` op.
5. **Dead-op sweep** at the end of the pass clears any orphaned `matlab.const_char` / `matlab.const_str` / `matlab.make_handle`.
6. **LLVM lowering** (existing pipeline) → `llvm.call` → bitcode → JIT.
7. **Runtime**: `matlab_plot2` records a Series; `matlab_set_series_linewidth` mutates the just-added series.

### When codegen rules out a call

If `LowerPlot` encounters a `matlab.call_builtin` whose name is in
`plotBuiltins()` but the operand pattern doesn't match any rewriter
branch, the rewrite returns `false`. The original `call_builtin` stays
intact and falls through to the next pass, which surfaces a verifier
error or a translate-to-LLVM-IR error. Common reasons:

- Argument types don't match what the runtime entry expects (e.g., a
  string literal where a `matlab_mat *` is required).
- Arity mismatch outside the supported set (e.g., `plot(x, y, z)`
  with 3 numeric args — neither 2-arg `plot` nor 3-arg `plot3`).

To diagnose, run `matlabc -emit-mlir <file>.m` and inspect the
`matlab.call_builtin` ops surviving past `LowerPlot`.

## 6. Quick-reference: enabling and using

### From a `.m` file via matlabc

```sh
cmake -B build -DMATLAB_LLVM_WITH_PLOT=ON -DMATLAB_LLVM_WITH_MLIR=ON
cmake --build build --target matlabc

cat <<'EOF' >/tmp/sin.m
x = 0:0.1:6;
y = sin(x);
figure;
plot(x, y, 'LineWidth', 2, 'Color', 'r');
title('sin(x)');
xlabel('x'); ylabel('y');
grid on;
print('/tmp/out.png');
EOF

cat /tmp/sin.m | build/matlabc -repl /dev/stdin
file /tmp/out.png    # → PNG image data, 800 x 600
```

### From C++ via the C ABI

```cpp
#include "matlab_plot.h"
#include "runtime_internal.h"

double xs[] = {0, 1, 2, 3, 4};
double ys[] = {0, 1, 4, 9, 16};
matlab_mat X{xs, 1, 5};
matlab_mat Y{ys, 1, 5};

matlab_figure_new();
matlab_plot2(&X, &Y);
matlab_title("squares", 7);
matlab_savefig("/tmp/sq.png", 11);
matlab_close_all();
```

The C ABI is stable across the matlab_plot library; codegen and tests
both consume it. JIT'd MATLAB programs end up making the same calls
through `DynamicLibrarySearchGenerator`.
