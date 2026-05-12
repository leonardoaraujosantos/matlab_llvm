#include "matlab/Sema/Resolver.h"
#include "matlab/Sema/Type.h"

#include <string>

namespace matlab {

Resolver::Resolver(SemaContext &Sema, TypeContext &TC, DiagnosticEngine &Diag)
    : Sema(Sema), TC(TC), Diag(Diag) {
  Global = Sema.newScope(nullptr, "<global>");
  registerBuiltins();
}

void Resolver::registerBuiltin(std::string_view Name) {
  Binding *B = Sema.newBinding();
  Global->declare(Name, BindingKind::Builtin, B);
}

void Resolver::registerBuiltins() {
  // Minimal initial registry. Type inference will special-case some of these
  // to produce concrete shape/dtype results.
  for (const char *N : {
    "zeros", "ones", "eye", "rand", "randn", "magic", "diag",
    "size", "length", "numel", "ndims",
    "reshape", "repmat", "linspace", "logspace",
    "abs", "sqrt", "exp", "log", "sin", "cos", "tan",
    "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
    "log2", "log10", "sign",
    "min", "max", "sum", "prod", "mean",
    "cumsum", "cumprod",
    "sort", "sortrows", "unique", "ismember",
    "setdiff", "intersect", "union",
    "horzcat", "vertcat", "permute", "squeeze",
    "flip", "fliplr", "flipud", "rot90",
    "sub2ind", "ind2sub", "assert",
    "mtimes", "mldivide", "mrdivide",
    "transpose", "ctranspose",
    "disp", "fprintf", "sprintf", "error", "warning", "input", "clear",
    "keyboard", "pause", "tic", "toc",
    "dbg", "who", "whos",
    "pi", "e", "Inf", "NaN", "eps", "realmin", "realmax",
    "isempty", "isequal", "find",
    "true", "false",
    "mod", "rem", "floor", "ceil", "round", "fix",
    "double", "single", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "logical", "char",
    /* Fixed-Point Designer (fi) — see docs/emit_fixed_point.md. */
    "fi", "numerictype", "fimath", "fipref",
    "int", "storedInteger", "storedIntegerToDouble",
    "reinterpretcast", "removefimath", "setfimath",
    "bin", "hex", "dec",
    "struct", "cell", "fieldnames", "isstruct", "isfield", "iscell",
    "rmfield", "string", "strlen", "isstring",
    /* Phase 4 — containers.Map / dictionary (key/value maps). */
    "containers", "dictionary", "isKey", "remove", "keys",
    /* Phase 5.1 — datetime / duration. */
    "datetime", "duration", "seconds", "minutes", "hours", "days", "years",
    "calendarDuration",
    /* Phase 5.2 — categorical. */
    "categorical", "iscategorical", "categories", "iscategory",
    /* Phase 5.3 — table. */
    "table", "istable", "height", "width", "rows2vars", "varfun",
    /* Phase 6 — Symbolic Math Toolbox via SymPP. The link target
     * (matlab_sym_* runtime) is only present when the build was
     * configured -DMATLAB_LLVM_WITH_SYM=ON; without that the JIT/-emit-c
     * paths will fail to resolve at link time, mirroring how a missing
     * BLAS would manifest. */
    "sym", "syms", "str2sym", "simplify", "expand", "factor", "subs",
    "solve", "vpa", "latex", "pretty", "ccode", "matlabFunction",
    "assume", "assumeAlso", "assumptions", "clearAssumptions",
    "taylor", "limit",
    "dsolve", "pdsolve", "pdsolve_heat", "pdsolve_wave",
    "laplace", "ilaplace", "fourier", "ifourier", "ztrans", "iztrans",
    "nsolve", "vpasolve", "checkodesol",
    "dsolve_ivp", "apply_ivp",
    /* Phase 6.1 — symbolic matrices. Constructed via the language-level
     * sym_matrix(rows, cols, e1, e2, ...) builtin since the standard
     * `[a 1; 2 b]` literal syntax doesn't yet detect sym entries to
     * route through matlab_symmat_new. */
    "sym_matrix", "sym_eye", "sym_zeros",
    "sym_det", "sym_inv", "sym_transpose", "sym_trace", "sym_rank",
    "sym_eigenvals", "sym_linsolve", "sym_dsolve_system",
    "sym_solve_sys", "sym_solve_2x2", "sym_solve_3x3",
    "sprintf", "num2str", "str2double",
    "upper", "lower", "startsWith", "endsWith", "contains",
    "strtrim", "strrep", "strcat",
    "svd", "eig", "inv", "pinv", "det", "rank", "cond",
    "qr", "lu", "chol", "norm", "trace", "kron",
    /* Tier 1.3 — Control System Toolbox roadmap §2.3 (matrix exponential
     * + matrix logarithm). */
    "expm", "logm",
    /* Tier 1.2 — Control System Toolbox roadmap §2.2 (Hessenberg reduction
     * + generalised Schur via the QZ algorithm). */
    "hess", "qz",
    /* Tier 1.2 follow-on — real Schur decomposition. */
    "schur",
    /* Tier 1.4 — Lyapunov / Stein equation solvers. */
    "lyap", "dlyap", "lyapchol",
    /* Tier 1.5 — algebraic Riccati. */
    "care", "dare", "icare", "idare",
    /* Tier 2 — first user-facing CST wrappers. lqi/lqry/lqg/kalman follow. */
    "lqr", "dlqr",
    /* Tier 3 — controllability/observability + pole placement. */
    "ctrb", "obsv", "place",
    /* Tier 3 — stability + characterization. */
    "isstable", "damp", "hsvd",
    /* Tier 4 — balancing + balanced truncation for model reduction.
     * `balred` (no suffix) is the user-facing 1-/3-return entry; the
     * `_A` / `_B` / `_C` variants are the splitter targets. */
    "balreal_T", "balred", "balred_A", "balred_B", "balred_C",
    /* Tier 3 — H₂ system norm (Lyapunov-based) + DC gain + stepinfo. */
    "norm_h2", "dcgain_ss", "stepinfo",
    /* Tier 4.2 — Kalman / Kalmd steady-state gains. `kalman` /
     * `kalmd` (no suffix) are the user-facing 1-/2-return entries; the
     * `_L` / `_P` variants are the splitter targets. */
    "kalman", "kalmd", "kalman_L", "kalmd_L",
    /* Tier 3 — discrete-time companions for stability + H₂ norm. */
    "isstable_d", "norm_h2_d",
    /* Tier 2.2 — continuous->discrete state-space ZOH + Tustin. */
    "c2d", "c2d_tustin",
    /* Inverse Tustin: discrete-to-continuous. */
    "d2c_tustin",
    /* Tier 3.4 / 2.3 — gramians and state-space step response. */
    "gram_c", "gram_o", "step_ss",
    /* Tier 2.4 — SISO state-space frequency response. */
    "bode_ss",
    /* Tier 2.3 follow-on + 2.4 — generalised SS sim, stability margins,
     * −3 dB bandwidth, peak gain (rough H∞), pole alias. */
    "lsim_ss", "gain_margin", "phase_margin", "bandwidth_ss",
    "getPeakGain_ss", "pole",
    /* Tier 2 — closed-loop assembly + interconnections (matrix-arg). */
    "feedback_ss", "series_ss", "parallel_ss", "append_ss",
    /* Tier 2.4 follow-on — TF (b, a) frequency response. */
    "bode_tf",
    /* §3.1 — model-object short forms (value-returning). Recognised
     * at the lowering site: a class-pinned first arg routes to the
     * matching matrix-arg primitive (e.g. `step(sys)` for an
     * ss-pinned `sys` → `step_ss(sys.A, sys.B, sys.C, sys.D, dt,
     * N)`). Class-returning short forms (`c2d(sys, Ts)`,
     * `feedback(sys1, sys2)`, …) are deferred — they need Sema to
     * pin the result slot to the matching class, which is out of
     * scope for the synthesised-builtin-call path today. Users
     * compose those explicitly via `ss(Ad, Bd, …)`. */
    "step", "bode", "dcgain", "lsim", "bandwidth",
    /* System Object lifecycle methods.  Registered as builtin names
     * so `reset(obj)` / `release(obj)` / `clone(obj)` / `isLocked(obj)`
     * resolve at Sema time; the generic method-on-class-instance
     * dispatch in Lowering.cpp routes the call to the matching
     * ClassName__<name> method body when the first arg is class-
     * pinned. */
    "reset", "release", "clone", "isLocked",
    /* MathWorks RF Propagation site methods — `pathloss(pm, rx, tx)`,
     * `los(tx, rx)`, `link(tx, rx)`, `sigstrength(rx, tx, pm)`,
     * `coverage(tx, pm, ...)`, `show(site)`.  Same function-style
     * dispatch on a class-pinned first arg. */
    "pathloss", "los", "link", "sigstrength", "coverage", "show",
    /* `siteviewer(...)` — MathWorks viewer launcher.  Text-only
     * stub here (no map); does nothing and returns 0. */
    "siteviewer",
    /* `design(antenna, freq)` — auto-resize a catalog antenna for
     * resonance at a given frequency.  Method-dispatch on the
     * antenna class. */
    "design",
    /* `antennaGain(antenna, freq)` — broadside peak gain in dBi.
     * Class-id dispatch in the runtime selects the textbook value
     * (dipole / monopole / etc.). */
    "antennaGain",
    /* §3.3 / §3.4 follow-ons — Tier-2 leftovers. impulse / initial
     * for time-domain free response; freqresp / nyquist / allmargin
     * for frequency-domain analysis. Matrix-arg companions
     * (impulse_ss / initial_ss / freqresp_ss / freqresp_tf /
     * nyquist_ss / nyquist_tf / allmargin_ss) ship alongside. */
    "impulse", "initial", "freqresp", "nyquist", "allmargin",
    "impulse_ss", "initial_ss", "freqresp_ss", "freqresp_tf",
    "nyquist_ss", "nyquist_tf", "allmargin_ss",
    /* §4 follow-ons — Tier-3 leftovers. `acker` = SISO Ackermann
     * (alias of place). `gram` / `norm` short forms select c/o
     * gramian or H₂ / H∞ norm via a char/scalar second arg.
     * `lqry` is output-weighted LQR. */
    "acker", "gram", "norm", "lqry",
    /* §5 follow-ons — Tier-4 leftovers. `pade` is the Padé time-
     * delay approximation [num, den] = pade(τ, n); `minreal` is the
     * tf-form pole-zero-cancellation [num_r, den_r] = minreal(num,
     * den, tol). Model-object short forms hsvd(sys) / balreal_T(sys)
     * route through Lowering.cpp's class-pinned-first-arg dispatch. */
    "pade", "minreal",
    /* §3.2 / §3.6 / §5.2 — class-returning model-object short forms.
     * Result is a fresh ss(.) constructor call; Resolver's
     * pinnedOfRhs propagates the class pin from the first arg. */
    "feedback", "series", "parallel", "append", "blkdiag",
    /* §5.1 / §5.3 — Tier-4 model-reduction + delay tail.
     * `sminreal` is the structural minimal realisation;
     * `modred` is modal residualisation/truncation; `thiran`
     * is the fractional-delay all-pass FIR builder. */
    "sminreal", "modred", "thiran",
    "arrayfun", "cellfun",
    /* Initial-value ODE solvers — see runtime/matlab_runtime.cpp.
     * ode23s is the Rosenbrock stiff solver. pdepe is the 1-D
     * parabolic-elliptic PDE solver via method-of-lines (uses
     * ode23s under the hood). */
    "ode45", "ode23", "ode23s", "ode_events", "pdepe",
    "nargin", "nargout", "varargin", "varargout",
    "fopen", "fclose", "fgetl", "feof",
    "fread", "fwrite",
    "readtable", "readmatrix",
    "save", "load",
    "conj", "real", "imag", "angle", "complex",
    "fft", "ifft", "fft2", "ifft2",
    "conv", "conv2",
    "filter", "any", "all", "tril", "triu",
    "fftshift", "ifftshift",
    "std", "var", "median", "diff",
    "meshgrid", "ndgrid",
    /* Tier 2 — signal/poly/numeric. */
    "xcorr", "polyval", "polyfit", "roots", "poly",
    "polyder", "polyint", "residue",
    /* Tier-1 §2.1 — IIR lowpass design + frequency response. */
    "butter", "cheby1", "cheby2", "freqz",
    "buttord", "cheb1ord", "cheb2ord",
    /* §2.1 follow-on — analog↔digital + form conversions + Bessel. */
    "bilinear", "freqs", "tf2zp", "zp2tf", "besself",
    "tf2sos", "sos2tf",
    /* Tier-1 §2.2 — FIR design + Savitzky-Golay. */
    "fir1", "sgolay", "sgolayfilt",
    /* Tier-1 §2.5 — close-the-loop filter helpers. */
    "filtfilt", "sosfilt", "impz", "stepz", "grpdelay",
    /* Tier-2 §3.4 — transforms tail. */
    "dct", "idct", "fwht", "hilbert", "goertzel",
    /* Tier-2 §3.1 — nonparametric spectral estimation. */
    "periodogram", "pwelch",
    /* Tier-2 §3.3 — time-frequency. */
    "spectrogram",
    /* Tier-2 §3.2 — linear prediction + parametric PSD. */
    "levinson", "lpc", "aryule", "arburg", "pyulear", "pburg",
    /* Tier-2 §3.1 cross-spectral helpers. */
    "cpsd", "mscohere", "tfestimate",
    /* Tier-3 §4.3 pulse measurements. */
    "findpeaks", "rms", "peak2peak", "peak2rms", "rssq",
    "medfilt1", "hampel", "envelope",
    "midcross", "risetime", "falltime", "dutycycle",
    "statelevels", "slewrate", "pulseperiod", "pulsewidth",
    "overshoot", "undershoot", "settlingtime",
    /* Tier-3 §4.1 multirate. */
    "upfirdn", "decimate", "interp", "resample",
    /* Tier-3 §4.2 waveform generators. */
    "chirp", "sawtooth", "square", "gauspuls",
    "rectpuls", "tripuls", "sinc",
    /* Tier-3 §4.4 alignment helpers. */
    "xcov", "finddelay", "dtw",
    "interp1", "trapz", "cumtrapz", "gradient",
    "hamming", "hann", "blackman",
    /* Tier-1 (Signal Processing Toolbox roadmap §2.3) — windows tail. */
    "rectwin", "triang", "bartlett", "barthannwin", "bohmanwin",
    "parzenwin", "nuttallwin", "blackmanharris", "flattopwin",
    "kaiser", "tukeywin", "gausswin", "chebwin", "taylorwin",
    /* Tier 3 — SVD-derived linalg + image-processing wrappers. */
    "null", "orth", "imfilter", "padarray",
    "interp2", "upsample", "downsample",
    /* Bitwise builtins. Lower to MATLAB's standard `&` / `|` / `^` /
     * `~` semantics on integer / fi-typed operands. The SV emitter
     * recognizes the matlab.call_builtin sites and renders them as
     * SV bitwise operators. */
    "bitand", "bitor", "bitxor", "bitcmp", "bitshift",
    /* Plotting (matlab_plot runtime; Cairo backend, headless cross-
     * platform PNG/SVG/PDF). Recognised by name in Sema; the lowering
     * pass LowerPlot.cpp rewrites matlab.call_builtin into LLVM calls
     * to matlab_figure_new / matlab_plot2 / matlab_plot_fmt /
     * matlab_title / etc. The runtime is built into matlabc when
     * configured with -DMATLAB_LLVM_WITH_PLOT=ON. */
    "figure", "gcf", "close",
    "plot", "plot3", "bar", "scatter", "stem", "stairs", "area",
    "errorbar", "histogram",
    "imshow", "imagesc", "pcolor", "surf", "mesh", "contour",
    "title", "xlabel", "ylabel", "zlabel", "legend", "text",
    "colorbar", "colormap",
    "grid", "hold", "axis", "box", "xlim", "ylim", "view",
    "loglog", "semilogx", "semilogy",
    "subplot", "saveas", "print",
    "xline", "yline",
    "xticks", "yticks", "xticklabels", "yticklabels",
    "yyaxis", "contourf", "quiver",
    /* PROP — Propagation Models (docs/comm_toolbox_roadmap.md §3).
     * Function-form, numeric-tag dispatch; runtime/runtime_prop.cpp. */
    "fspl",
    "pathlossHata", "pathlossCost231", "pathlossEgli", "pathlossEcc33",
    "pathlossSui", "pathlossEricsson9999",
    "pathlossRain", "pathlossGas", "pathlossFog", "pathlossCloseIn",
    "fresnelZoneRadius", "fresnelClearance",
    "diffractionKnifeEdge", "diffractionBullington", "diffractionDeygout",
    "haversine", "bearing", "vincenty",
    "greatCircleDestLat", "greatCircleDestLon",
    "itmPathloss",
    "terrainProfile", "losObstruction", "losClear",
    "linkBudget", "coverageGrid",
    "sectorPattern", "cosinePattern", "gaussianPattern", "isotropicPattern",
    "applyMountOrientation", "applyMountAz", "applyMountEl",
    "coverageGridMulti",
    /* PROP §3.5 — PropagationModel classdef dispatcher.  Called
     * from the `pathloss(pm, rx, tx)` method body in
     * `runtime/rf_class_propagationmodel.m`. */
    "propPathlossDispatch",
    "propLosSites",
    "propKindToModelCode",
    /* COMM Tier-1 (docs/comm_toolbox_roadmap.md §2). Function-form
     * base layer; runtime/runtime_comm.cpp. Numeric tag dispatch.
     * `rng(seed)` uses the existing `rng` name (also already
     * registered above by no other entry — only `rngDefault` /
     * `rngShuffle` / `rngGet` / `rngSet` add new names). */
    "randi", "rng", "rngDefault", "rngShuffle", "rngGet", "rngSet",
    "randsrc", "randsrcWeighted", "randerr",
    "int2bit", "bit2int", "de2bi", "bi2de",
    "awgn",
    "biterr", "biterrK", "biterrCount", "symerr", "symerrCount",
    /* COMM Tier-2 — digital modulation MVP (docs/comm_toolbox_roadmap.md §4).
     * Numeric tag dispatch: order code 0=binary, 1=Gray; output code
     * 0=integer-hard, 1=bit, 2=LLR; modulation code 0=PAM, 1=PSK,
     * 2=QAM, 3=DPSK, 4=FSK-coh, 5=FSK-nc; rcosdesign shape 0=sqrt
     * (RRC), 1=normal (full RC). */
    "pammod", "pamdemod", "pskmod", "pskdemod",
    "qammod", "qamdemod", "qamdemodBit", "qamdemodLlr",
    "genqammod", "genqamdemod",
    "rcosdesign", "gaussdesign",
    "berawgn", "scatterplot", "eyediagram",
    "qfunc", "erfc",
    /* COMM Tier-3 — channel coding (docs/comm_toolbox_roadmap.md §5).
     * Function-form CRC + convolutional + Hamming + interleavers.
     * BCH / RS / gf + LDPC / Turbo / Polar are deferred. */
    "crcGenerate", "crcCheck", "crcStrip",
    "poly2trellis", "convenc", "vitdec", "oct2dec",
    "hammgenParity", "hammingEncode", "hammingDecode",
    "intrlv", "deintrlv",
    /* COMM Tier-4 — equalisation, sync, RF impairments
     * (docs/comm_toolbox_roadmap.md §6).  Function-form. */
    "lms", "rls", "cma", "dfe",
    "costasPll", "symbolSyncMM", "preambleDetect",
    "phaseFreqOffset", "iqimbal", "memorylessNl", "phaseNoise",
    "vitdecSoft",
    /* COMM Tier-5 — OFDM / fading / MIMO
     * (docs/comm_toolbox_roadmap.md §7).  Function-form. */
    "ofdmmod", "ofdmdemod",
    "rayleighChannel", "ricianChannel",
    "ostbcEncode", "ostbcCombine", "mlDetect",
    /* COMM Tier-6 — spreading + source coding
     * (docs/comm_toolbox_roadmap.md §8). Function-form. */
    "pnSequence", "goldSequence", "hadamard", "walshCode",
    "quantiz", "quantizApply", "lloydsQuant",
    "compandMu", "compandA",
    "dpcmEncode", "dpcmDecode",
    /* COMM Tier-7 — modern codes
     * (docs/comm_toolbox_roadmap.md §5.4).  Function-form. */
    "polarEncode", "polarSCdecode",
    "ldpcEncode",  "ldpcDecodeMS",
    "turboEncode", "turboDecode",
    /* RF Toolbox companion (docs/comm_toolbox_roadmap.md §9).
     * Function-form 2-port subset: Touchstone I/O + closed-form
     * S-parameter analyses + Friis cascade. */
    "touchstoneRead", "touchstoneWriteS2p",
    "tsS11", "tsS12", "tsS21", "tsS22",
    "tsFreqs", "tsZ0", "tsNumPorts",
    "tsSij", "tsYij", "tsZij", "tsHij", "tsGij", "tsTij",
    "tsAbcdA", "tsAbcdB", "tsAbcdC", "tsAbcdD",
    "sparamS2yN", "sparamS2zN", "snp2smp", "snp2smpZ",
    "touchstoneWrite", "cascadeSparamsN", "cascadeSparamsNFull",
    "sparamS2h", "sparamS2abcd",
    "sparamH2s", "sparamAbcd2s",
    "sparamS2g", "sparamG2s", "sparamS2t", "sparamT2s",
    "gamma2z", "z2gamma",
    "matchingnetworkT", "matchingnetworkPi",
    "sparamS2y", "sparamS2z",
    "gammaIn", "gammaOut", "vswr",
    "powerGain", "stabilityK", "stabilityMu", "s2tf",
    "cascadeSparams2", "rfbudgetFriis",
    /* RF-Tier-3.1 — Vector Fitting + freqresp on the fitted model. */
    "rationalfit", "freqresp",
    "rfPoles", "rfResidues", "rfD", "rfOrder", "rfFitError",
    /* RF-Tier-3.2 — time-domain RF (timeresp + step-driven TDR/TDT). */
    "timeresp", "s2tdr", "s2tdt",
    /* RF-Tier-2 mixed-mode 4-port + Smith chart grid + passivity. */
    "sparamS2smm", "smithGrid", "smithRCircle", "smithUnitCircle", "passivity",
    /* RF-Tier-4.1 matchingnetwork L-section auto-synthesis. */
    "matchingnetwork",
    /* RF-Tier-3.3 transmission-line geometries. */
    "rfckt_txline", "rfckt_coaxial", "rfckt_microstrip",
    "rfckt_cpw", "rfckt_parallelplate", "rfckt_twowire",
    "rfckt_lcfilter", "rfckt_lcfilter4",
    "rfAnalyzeAmplifier", "rfAnalyzePassive",
    "rfAnalyzeSeries", "rfAnalyzeShunt",
    "gammams", "gammaml", "groupdelay",
    "s2tfPort", "rfbudgetTable",
    "stabCircleLoad", "stabCircleSource",
    /* MathWorks-faithful lowercase aliases. */
    "s2y", "s2z", "s2h", "s2g", "s2abcd", "s2t",
    "h2s", "g2s", "abcd2s", "t2s",
    "rfbudget", "rfwrite", "sparameters",
    /* RF-Tier-3.1 follow-on — rationalfit delay + passivity. */
    "rfDelayEstimate", "rfApplyDelay", "rfPassivityEnforce",
    "rationalfitWeighted", "newref", "cascadeSparamsNFullK",
    "sparamS2abcdN", "sparamS2hN",
    /* RF-Tier-5 — Verilog-A export. */
    "writeVerilogA", "writeVerilogATF", "writeVerilogAZPK",
    "writeVerilogASS",
  }) {
    registerBuiltin(N);
  }
}

Binding *Resolver::declareFn(Scope *S, Function *F) {
  Binding *B = Sema.newBinding();
  Binding *Declared = S->declare(F->Name, BindingKind::Function, B);
  Declared->FuncDef = F;
  F->Self = Declared;
  return Declared;
}

Binding *Resolver::declareClass(Scope *S, ClassDef *C) {
  Binding *B = Sema.newBinding();
  Binding *Declared = S->declare(C->Name, BindingKind::Class, B);
  Declared->ClassDef = C;
  C->Self = Declared;
  return Declared;
}

void Resolver::resolve(TranslationUnit &TU) {
  // Register all top-level functions in the global scope *before* resolving
  // their bodies so mutual calls work.
  for (Function *F : TU.Functions) declareFn(Global, F);
  /* Classes share the same global name space as functions — `ClassName(args)`
   * looks like a function call at the parser level and is disambiguated
   * here by binding kind. Register before resolving any bodies so
   * script-level constructor calls resolve. */
  int32_t NextClassId = 1;
  for (ClassDef *C : TU.Classes) {
    declareClass(Global, C);
    C->ClassId = NextClassId++;
    /* Give each property a stable id per class (used by the runtime's
     * property table so names don't have to be re-hashed at every
     * access). */
    int32_t PropId = 0;
    for (auto &P : C->Props) P.PropId = PropId++;
  }
  /* Resolve superclass pointers. `handle` is MATLAB's root reference-
   * semantics class and has no user-facing behavior in our runtime —
   * we accept the syntax but leave Super = null. Any other name must
   * resolve to another classdef declared in this TU. */
  for (ClassDef *C : TU.Classes) {
    if (C->SuperName.empty() || C->SuperName == "handle") continue;
    Binding *SB = Global->lookup(C->SuperName);
    if (SB && SB->Kind == BindingKind::Class && SB->ClassDef) {
      C->Super = SB->ClassDef;
    } else {
      Diag.error(C->Range.Begin,
                 std::string("unknown superclass '") +
                     std::string(C->SuperName) + "' for class '" +
                     std::string(C->Name) + "'");
    }
  }

  if (TU.ScriptNode) {
    Scope *ScriptScope = Sema.newScope(Global, "<script>");
    // Pre-collect assignments so NameExpr uses can see script-local vars.
    collectAssignmentsInBlock(*TU.ScriptNode->Body, ScriptScope);
    resolveBlock(*TU.ScriptNode->Body, ScriptScope);
  }

  for (Function *F : TU.Functions) resolveFunction(*F, Global);

  /* Resolve class method bodies. Each method is an ordinary Function
   * whose first parameter (`obj`) is typed as the owning class, so
   * property accesses route correctly. Operator-overload methods that
   * take a second class operand (e.g. plus(a, b)) reach the right
   * dispatch path via matlab_obj's struct-compatible layout — the
   * field-access helpers work whether the caller went through
   * matlab_obj_* or matlab_struct_*. */
  auto isBinaryObjectOperator = [](std::string_view N) {
    return N == "plus" || N == "minus" ||
           N == "eq" || N == "ne" || N == "lt" || N == "le" ||
           N == "gt" || N == "ge" ||
           N == "and" || N == "or";
  };
  /* CST prelude classes (tf / ss / zpk / pid / frd) overload the
   * scalar-mixing ops `mtimes` / `mrdivide` / etc. with both operands
   * being class instances — a `tf * tf` series-cascade or `tf / tf`
   * inversion never sees a scalar second argument, so we want the
   * resolver to pin the second param to the class so its body reads
   * (`b.Numerator`, `b.Denominator`) route through the class path.
   * Other user classes (Vec2, BasicClass, …) keep the historical
   * scalar-mixing behaviour for those operators. The list is the same
   * one matlabc/main.cpp's `userMentionsCstClass` looks for. */
  auto isCstClass = [](std::string_view N) {
    return N == "tf" || N == "ss" || N == "zpk" || N == "pid" ||
           N == "frd";
  };
  auto isExtendedBinaryObjectOperator = [](std::string_view N) {
    return N == "mtimes" || N == "times" ||
           N == "mrdivide" || N == "rdivide" ||
           N == "mldivide" || N == "ldivide" ||
           N == "mpower" || N == "power";
  };
  for (ClassDef *C : TU.Classes) {
    for (Function *M : C->Methods) {
      resolveFunction(*M, Global);
      /* Constructor: `function obj = ClassName(args)`. The obj is an
       * Output — first Input is an ordinary user arg, not the self
       * pointer. Pin the Output here; the first Input is left alone.
       * Non-constructor: first Input is the self `obj` — pin it. */
      bool IsCtor = M->Name == C->Name;
      if (IsCtor) {
        if (!M->OutputRefs.empty() && M->OutputRefs.front())
          M->OutputRefs.front()->PinnedClass = C;
      } else {
        if (!M->ParamRefs.empty() && M->ParamRefs.front())
          M->ParamRefs.front()->PinnedClass = C;
      }
      /* For binary-object operators, also pin the second param — both
       * operands are expected to be the same class in those cases,
       * and pinning lets property reads route through the class path
       * even when the method body uses `b.field`. The unconditional
       * list (plus / minus / eq / ne / …) covers any user classdef.
       * The extended list (mtimes / times / mrdivide / mpower / …)
       * is gated on the CST prelude allowlist so user classes that
       * scalar-mix in those operators (Vec2, …) keep working. */
      if (!IsCtor && M->ParamRefs.size() >= 2 && M->ParamRefs[1]) {
        bool BasePin = isBinaryObjectOperator(M->Name);
        bool CstPin = isCstClass(C->Name) &&
                      isExtendedBinaryObjectOperator(M->Name);
        if (BasePin || CstPin)
          M->ParamRefs[1]->PinnedClass = C;
      }
    }
    for (Function *M : C->StaticMethods) {
      resolveFunction(*M, Global);
    }
  }

  /* Inter-procedural class pinning propagation.
   *
   * Sema's per-method param pinning (above) covers the SELF param
   * (always class C for non-ctor methods on C) and the second
   * operand of binary-operator overloads.  Other method params
   * stay unpinned until a call site informs them.
   *
   * Without propagation, a method body like
   *
   *     function ss = sigstrength(rx, tx, pm)
   *         pl = pathloss(pm, rx, tx);   % pm unpinned → fall-through
   *         ...
   *
   * can't route `pathloss(pm, ...)` through the method-on-class
   * dispatch (which keys on PinnedClass of the first arg).  When
   * the user calls `sigstrength(rx, tx, pm)` with pm pinned to
   * PropagationModel, the pin should propagate to the callee's
   * `pm` parameter.
   *
   * Algorithm:
   *   For each CallOrIndex in every script / function / method body,
   *   look up the callee Function* (direct user function or class
   *   method via builtin-name + first-arg class).  For each arg
   *   that's a NameExpr with PinnedClass set, propagate to the
   *   matching callee param Binding (when its PinnedClass is null).
   *
   *   Iterate to fixpoint: a newly-pinned param may enable more
   *   pinning at calls inside its method body.
   *
   * The pin is read at lowering time (FieldAccess / CallOrIndex
   * dispatch sites both consult Binding->PinnedClass), so setting
   * it post-resolve still affects codegen. */
  /* Recursive class-pin extractor — same shape as the existing
   * `pinnedOfRhs` for LHS-pinning above, but exposed here as a
   * local helper so the IP propagation can ask "is this arg
   * class-pinned?" for ANY expression shape (NameExpr,
   * ctor call, method-style call returning a class, etc.). */
  std::function<const ClassDef *(const Expr *)> argPin =
      [&argPin](const Expr *RE) -> const ClassDef * {
    if (!RE) return nullptr;
    if (auto *NE = dynamic_cast<const NameExpr *>(RE)) {
      return NE->Ref ? NE->Ref->PinnedClass : nullptr;
    }
    if (auto *CX = dynamic_cast<const CallOrIndex *>(RE)) {
      if (auto *NX = dynamic_cast<const NameExpr *>(CX->Callee)) {
        /* Direct constructor call: ClassName(args) → that class. */
        if (NX->Ref && NX->Ref->Kind == BindingKind::Class &&
            NX->Ref->ClassDef) return NX->Ref->ClassDef;
        /* Function-style class-method dispatch returning a class
         * instance — `design(antenna, freq)` returns an AntDipole,
         * `clone(obj)` returns the same class.  Resolve via the
         * first-arg class + method-name lookup, then read the
         * method's first output's PinnedClass. */
        if (NX->Ref && NX->Ref->Kind == BindingKind::Builtin &&
            !CX->Args.empty()) {
          const ClassDef *FirstCls = argPin(CX->Args[0]);
          if (FirstCls) {
            for (const ClassDef *CC = FirstCls; CC; CC = CC->Super) {
              for (Function *M : CC->Methods) {
                if (!M || M->Name != NX->Name) continue;
                if (!M->OutputRefs.empty() && M->OutputRefs.front() &&
                    M->OutputRefs.front()->PinnedClass)
                  return M->OutputRefs.front()->PinnedClass;
              }
            }
          }
        }
      }
    }
    return nullptr;
  };
  auto propagateOne = [&](const CallOrIndex *C) -> bool {
    if (!C || !C->Callee || C->Args.empty()) return false;
    auto *N = dynamic_cast<const NameExpr *>(C->Callee);
    if (!N || !N->Ref) return false;
    Function *Callee = nullptr;
    /* Direct function call: callee binding points at the function. */
    if (N->Ref->Kind == BindingKind::Function && N->Ref->FuncDef) {
      Callee = N->Ref->FuncDef;
    } else if (N->Ref->Kind == BindingKind::Builtin) {
      /* Function-style class-method dispatch: when the first arg is
       * class-pinned and the class (or any ancestor) has a method
       * with the same name as the callee, the lowering routes there.
       * Mirror the same lookup here so pinning propagates. */
      const ClassDef *FirstCls = argPin(C->Args[0]);
      if (FirstCls) {
        for (const ClassDef *CC = FirstCls; CC; CC = CC->Super) {
          for (Function *M : CC->Methods)
            if (M && M->Name == N->Name) { Callee = M; break; }
          if (Callee) break;
        }
      }
    }
    if (!Callee) return false;
    bool Changed = false;
    size_t NArgs = C->Args.size();
    if (getenv("MATLABC_DBG_IP_PIN")) {
      fprintf(stderr, "[ip-pin] %s(", std::string(N->Name).c_str());
      for (size_t i = 0; i < NArgs; ++i) {
        const ClassDef *Pin = argPin(C->Args[i]);
        const char *p = Pin ? std::string(Pin->Name).c_str() : "-";
        fprintf(stderr, "%s%s", i==0?"":",", p);
      }
      fprintf(stderr, ") → %s\n", std::string(Callee->Name).c_str());
    }
    for (size_t i = 0; i < NArgs && i < Callee->ParamRefs.size(); ++i) {
      if (!C->Args[i]) continue;
      const ClassDef *Pin = argPin(C->Args[i]);
      if (!Pin) continue;
      Binding *PB = Callee->ParamRefs[i];
      if (!PB || PB->PinnedClass) continue;
      PB->PinnedClass = const_cast<ClassDef *>(Pin);
      Changed = true;
    }
    return Changed;
  };
  std::function<void(const Expr &, bool &)> walkExpr;
  std::function<void(const Stmt &, bool &)> walkStmt;
  std::function<void(const Block &, bool &)> walkBlock;
  walkExpr = [&](const Expr &E, bool &Changed) {
    if (auto *C = dynamic_cast<const CallOrIndex *>(&E)) {
      if (propagateOne(C)) Changed = true;
      if (C->Callee) walkExpr(*C->Callee, Changed);
      for (Expr *A : C->Args) if (A) walkExpr(*A, Changed);
      return;
    }
    if (auto *Bi = dynamic_cast<const BinaryOpExpr *>(&E)) {
      if (Bi->LHS) walkExpr(*Bi->LHS, Changed);
      if (Bi->RHS) walkExpr(*Bi->RHS, Changed);
      return;
    }
    if (auto *U = dynamic_cast<const UnaryOpExpr *>(&E)) {
      if (U->Operand) walkExpr(*U->Operand, Changed);
      return;
    }
    if (auto *F = dynamic_cast<const FieldAccess *>(&E)) {
      if (F->Base) walkExpr(*F->Base, Changed);
      return;
    }
    if (auto *M = dynamic_cast<const MatrixLiteral *>(&E)) {
      for (auto &Row : M->Rows)
        for (Expr *X : Row) if (X) walkExpr(*X, Changed);
      return;
    }
  };
  /* Propagate method-return class pin back to caller LHS.  When
   * `d = design(AntDipole(), f)` is processed, the script body
   * resolves before the method bodies, so at script-resolve time
   * `design`'s OutputRefs[0]->PinnedClass is null.  After all
   * method bodies are resolved (and the inner `r = AntDipole(...)`
   * pinned `r` to AntDipole), this pass re-runs pinnedOfRhs on
   * AssignStmt RHS expressions and updates the LHS binding's
   * PinnedClass accordingly. */
  auto propagateLhsPin = [&](const AssignStmt *A, bool &Changed) {
    if (!A || !A->RHS) return;
    const ClassDef *Pin = argPin(A->RHS);
    if (!Pin) return;
    for (Expr *L : A->LHS) {
      if (auto *LN = dynamic_cast<NameExpr *>(L)) {
        if (LN->Ref && !LN->Ref->PinnedClass &&
            LN->Ref->Kind != BindingKind::Class) {
          LN->Ref->PinnedClass = const_cast<ClassDef *>(Pin);
          Changed = true;
        }
      }
    }
  };
  walkStmt = [&](const Stmt &S, bool &Changed) {
    if (auto *A = dynamic_cast<const AssignStmt *>(&S)) {
      propagateLhsPin(A, Changed);
      for (Expr *L : A->LHS) if (L) walkExpr(*L, Changed);
      if (A->RHS) walkExpr(*A->RHS, Changed);
      return;
    }
    if (auto *E = dynamic_cast<const ExprStmt *>(&S)) {
      if (E->E) walkExpr(*E->E, Changed);
      return;
    }
    if (auto *I = dynamic_cast<const IfStmt *>(&S)) {
      if (I->Cond) walkExpr(*I->Cond, Changed);
      if (I->Then) walkBlock(*I->Then, Changed);
      for (auto &EI : I->Elseifs) {
        if (EI.Cond) walkExpr(*EI.Cond, Changed);
        if (EI.Body) walkBlock(*EI.Body, Changed);
      }
      if (I->Else) walkBlock(*I->Else, Changed);
      return;
    }
    if (auto *F = dynamic_cast<const ForStmt *>(&S)) {
      if (F->Iter) walkExpr(*F->Iter, Changed);
      if (F->Body) walkBlock(*F->Body, Changed);
      return;
    }
    if (auto *W = dynamic_cast<const WhileStmt *>(&S)) {
      if (W->Cond) walkExpr(*W->Cond, Changed);
      if (W->Body) walkBlock(*W->Body, Changed);
      return;
    }
    if (auto *Sw = dynamic_cast<const SwitchStmt *>(&S)) {
      if (Sw->Discriminant) walkExpr(*Sw->Discriminant, Changed);
      for (auto &Cs : Sw->Cases) {
        if (Cs.Value) walkExpr(*Cs.Value, Changed);
        if (Cs.Body) walkBlock(*Cs.Body, Changed);
      }
      return;
    }
  };
  walkBlock = [&](const Block &B, bool &Changed) {
    for (Stmt *S : B.Stmts) if (S) walkStmt(*S, Changed);
  };
  /* Fixpoint — re-walk until no new pin is propagated.  Bounded
   * iteration count just in case (the propagation is monotonic so
   * fixpoint is finite, but the cap protects against accidental
   * regressions). */
  for (int Iter = 0; Iter < 16; ++Iter) {
    bool Changed = false;
    if (TU.ScriptNode && TU.ScriptNode->Body)
      walkBlock(*TU.ScriptNode->Body, Changed);
    for (Function *F : TU.Functions)
      if (F->Body) walkBlock(*F->Body, Changed);
    for (ClassDef *C : TU.Classes) {
      for (Function *M : C->Methods) if (M->Body) walkBlock(*M->Body, Changed);
      for (Function *M : C->StaticMethods) if (M->Body) walkBlock(*M->Body, Changed);
    }
    if (!Changed) break;
  }
}

//===----------------------------------------------------------------------===//
// Assignment collection (pre-pass to populate variable bindings).
//===----------------------------------------------------------------------===//

void Resolver::collectAssignments(Function &F, Scope *FnScope) {
  // Parameters and outputs are declared up-front.
  for (auto Name : F.Inputs) {
    if (Name == "~") continue; // placeholder parameter
    Binding *B = Sema.newBinding();
    Binding *D = FnScope->declare(Name, BindingKind::Param, B);
    F.ParamRefs.push_back(D);
  }
  for (auto Name : F.Outputs) {
    Binding *B = Sema.newBinding();
    Binding *D = FnScope->declare(Name, BindingKind::Output, B);
    F.OutputRefs.push_back(D);
  }
  // Register nested functions so calls to them resolve inside the parent body.
  for (Function *N : F.Nested) {
    declareFn(FnScope, N);
  }
  if (F.Body) collectAssignmentsInBlock(*F.Body, FnScope);
}

void Resolver::collectAssignmentsInBlock(Block &B, Scope *FnScope) {
  for (Stmt *S : B.Stmts)
    if (S) collectAssignmentsInStmt(*S, FnScope);
}

void Resolver::collectAssignmentsInStmt(Stmt &S, Scope *FnScope) {
  switch (S.Kind) {
  case NodeKind::AssignStmt: {
    auto &A = static_cast<AssignStmt &>(S);
    for (Expr *L : A.LHS) {
      // Peel off indexing/field-access to find the root name.
      Expr *Root = L;
      while (Root) {
        switch (Root->Kind) {
        case NodeKind::NameExpr: {
          auto *N = static_cast<NameExpr *>(Root);
          Binding *B = Sema.newBinding();
          FnScope->getOrDeclareVar(N->Name, B);
          Root = nullptr;
          break;
        }
        case NodeKind::CallOrIndex:
          Root = static_cast<CallOrIndex *>(Root)->Callee;
          break;
        case NodeKind::CellIndex:
          Root = static_cast<CellIndex *>(Root)->Callee;
          break;
        case NodeKind::FieldAccess:
          Root = static_cast<FieldAccess *>(Root)->Base;
          break;
        case NodeKind::DynamicField:
          Root = static_cast<DynamicField *>(Root)->Base;
          break;
        default:
          Root = nullptr;
          break;
        }
      }
    }
    break;
  }
  case NodeKind::ForStmt: {
    auto &F = static_cast<ForStmt &>(S);
    if (!F.Var.empty()) {
      Binding *B = Sema.newBinding();
      /* A prior for-loop with the same variable name reuses its
       * binding; keep VarRef pointing at the real one so the lowerer's
       * slot lookup returns the shared slot. */
      F.VarRef = FnScope->getOrDeclareVar(F.Var, B);
    }
    if (F.Body) collectAssignmentsInBlock(*F.Body, FnScope);
    break;
  }
  case NodeKind::WhileStmt: {
    auto &W = static_cast<WhileStmt &>(S);
    if (W.Body) collectAssignmentsInBlock(*W.Body, FnScope);
    break;
  }
  case NodeKind::IfStmt: {
    auto &I = static_cast<IfStmt &>(S);
    if (I.Then) collectAssignmentsInBlock(*I.Then, FnScope);
    for (auto &EI : I.Elseifs)
      if (EI.Body) collectAssignmentsInBlock(*EI.Body, FnScope);
    if (I.Else) collectAssignmentsInBlock(*I.Else, FnScope);
    break;
  }
  case NodeKind::SwitchStmt: {
    auto &Sw = static_cast<SwitchStmt &>(S);
    for (auto &C : Sw.Cases)
      if (C.Body) collectAssignmentsInBlock(*C.Body, FnScope);
    break;
  }
  case NodeKind::TryStmt: {
    auto &T = static_cast<TryStmt &>(S);
    if (T.TryBody) collectAssignmentsInBlock(*T.TryBody, FnScope);
    if (!T.CatchVar.empty()) {
      Binding *B = Sema.newBinding();
      T.CatchVarRef = FnScope->getOrDeclareVar(T.CatchVar, B);
    }
    if (T.CatchBody) collectAssignmentsInBlock(*T.CatchBody, FnScope);
    break;
  }
  case NodeKind::GlobalDecl: {
    auto &G = static_cast<GlobalDecl &>(S);
    for (auto N : G.Names) {
      Binding *B = Sema.newBinding();
      FnScope->declare(N, BindingKind::Global, B);
    }
    break;
  }
  case NodeKind::PersistentDecl: {
    auto &P = static_cast<PersistentDecl &>(S);
    for (auto N : P.Names) {
      Binding *B = Sema.newBinding();
      FnScope->declare(N, BindingKind::Persistent, B);
    }
    break;
  }
  case NodeKind::CommandStmt: {
    /* Most commands don't introduce variables — but `syms x y z` does,
     * matching MATLAB's Symbolic Math Toolbox semantics where the
     * names are declared as fresh symbolic variables in the current
     * workspace. Treat each argument as a variable name and declare
     * it; lowering will emit matlab_sym_named + matlab_ws_set_sym. */
    auto &C = static_cast<CommandStmt &>(S);
    if (C.Name == "syms") {
      for (auto &Arg : C.Args) {
        /* Skip option-like tokens MATLAB accepts (real, integer,
         * positive, ...) — those are assumption modifiers handled in
         * Phase B. Heuristic: an argument is a name iff it's a
         * MATLAB-shaped identifier. */
        if (Arg.empty()) continue;
        char c = Arg.front();
        bool isIdent = (c == '_' ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= 'a' && c <= 'z'));
        if (!isIdent) continue;
        Binding *B = Sema.newBinding();
        FnScope->getOrDeclareVar(Arg, B);
      }
    }
    break;
  }
  default:
    break;
  }
}

//===----------------------------------------------------------------------===//
// Resolution pass.
//===----------------------------------------------------------------------===//

void Resolver::resolveFunction(Function &F, Scope *Parent) {
  F.FnScope = Sema.newScope(Parent, std::string(F.Name));
  collectAssignments(F, F.FnScope);
  if (F.Body) resolveBlock(*F.Body, F.FnScope);
  for (Function *N : F.Nested) resolveFunction(*N, F.FnScope);
}

void Resolver::resolveBlock(Block &B, Scope *S) {
  for (Stmt *St : B.Stmts) if (St) resolveStmt(*St, S);
}

void Resolver::resolveStmt(Stmt &St, Scope *S) {
  switch (St.Kind) {
  case NodeKind::ExprStmt: {
    auto &E = static_cast<ExprStmt &>(St);
    if (E.E) resolveExpr(*E.E, S);
    break;
  }
  case NodeKind::AssignStmt: {
    auto &A = static_cast<AssignStmt &>(St);
    if (A.RHS) resolveExpr(*A.RHS, S);
    for (Expr *L : A.LHS) resolveLValue(*L, S);
    /* Pin the LHS variable to the class of the RHS when the RHS is a
     * direct constructor call `ClassName(args)`. Later lookups of
     * `lhs.prop` or `lhs.method(args)` then dispatch against this class
     * without dynamic type discovery. */
    /* Walk the RHS to find a class hint. A direct ClassName(args)
     * constructor call obviously produces an instance of that class.
     * A BinaryOp where either operand is pinned to a class is treated
     * as producing another instance of that class (the operator
     * overload's assumed return type for arithmetic ops). Similarly a
     * dot-method call `obj.m(args)` on a pinned obj returns... we
     * don't know, so skip. Returning a new instance is the common
     * pattern for v1 and matches the BasicClass / Vec2 examples. */
    std::function<ClassDef *(Expr *)> pinnedOfRhs =
        [&pinnedOfRhs](Expr *RE) -> ClassDef * {
      if (!RE) return nullptr;
      if (auto *NE = dynamic_cast<NameExpr *>(RE)) {
        if (NE->Ref && NE->Ref->PinnedClass) return NE->Ref->PinnedClass;
        return nullptr;
      }
      if (auto *CX = dynamic_cast<CallOrIndex *>(RE)) {
        if (auto *NX = dynamic_cast<NameExpr *>(CX->Callee)) {
          if (NX->Ref && NX->Ref->Kind == BindingKind::Class &&
              NX->Ref->ClassDef) return NX->Ref->ClassDef;
          /* Function-style class-method dispatch returning a fresh
           * class instance — e.g. `design(antenna, freq)` returning
           * an AntDipole.  When the callee is a builtin name AND
           * the first arg is class-pinned AND that class has a
           * method with the matching name AND the method's first
           * output binding is class-pinned, propagate that pin.
           * Covers `design(AntDipole, f)` / `clone(obj)` / etc. */
          if (NX->Ref && NX->Ref->Kind == BindingKind::Builtin &&
              !CX->Args.empty()) {
            ClassDef *Arg0Cls = pinnedOfRhs(CX->Args[0]);
            if (Arg0Cls) {
              for (ClassDef *CC = Arg0Cls; CC; CC = CC->Super) {
                for (Function *M : CC->Methods) {
                  if (!M || M->Name != NX->Name) continue;
                  if (!M->OutputRefs.empty() && M->OutputRefs.front() &&
                      M->OutputRefs.front()->PinnedClass)
                    return M->OutputRefs.front()->PinnedClass;
                  /* Method exists but output isn't pinned — fall
                   * through.  The CST-special-case block below
                   * handles `c2d(sys, Ts)` etc. */
                }
              }
            }
          }
          /* §3.1 / Tier-4 — class-returning model-object short forms.
           * When the first arg is class-pinned and the callee is a
           * known short-form name, the result is the same class.
           * Mirrors the Lowering.cpp class-pinned-first-arg dispatch
           * that emits the corresponding constructor call. */
          if (NX->Ref && NX->Ref->Kind == BindingKind::Builtin &&
              !CX->Args.empty()) {
            ClassDef *Arg0Cls = pinnedOfRhs(CX->Args[0]);
            if (Arg0Cls && Arg0Cls->Name == "ss") {
              const auto &Nm = NX->Name;
              if (Nm == "c2d" || Nm == "c2d_tustin" ||
                  Nm == "d2c_tustin" || Nm == "feedback" ||
                  Nm == "series" || Nm == "parallel" ||
                  Nm == "append" || Nm == "blkdiag" ||
                  Nm == "sminreal" || Nm == "modred")
                return Arg0Cls;
            }
          }
        }
      }
      if (auto *Bi = dynamic_cast<BinaryOpExpr *>(RE)) {
        if (auto *L = pinnedOfRhs(Bi->LHS)) {
          bool IsCmp =
              Bi->Op == BinOp::Eq || Bi->Op == BinOp::Ne ||
              Bi->Op == BinOp::Lt || Bi->Op == BinOp::Le ||
              Bi->Op == BinOp::Gt || Bi->Op == BinOp::Ge;
          if (!IsCmp) return L;
        }
        if (auto *R = pinnedOfRhs(Bi->RHS)) {
          bool IsCmp =
              Bi->Op == BinOp::Eq || Bi->Op == BinOp::Ne ||
              Bi->Op == BinOp::Lt || Bi->Op == BinOp::Le ||
              Bi->Op == BinOp::Gt || Bi->Op == BinOp::Ge;
          if (!IsCmp) return R;
        }
      }
      /* Unary class-method overloads (`-tf_obj`, `+tf_obj`, etc.)
       * return a fresh class instance — propagate the pin through
       * the unary op so the LHS slot picks up the class. */
      if (auto *U = dynamic_cast<UnaryOpExpr *>(RE)) {
        if (auto *L = pinnedOfRhs(U->Operand)) return L;
      }
      return nullptr;
    };
    if (ClassDef *RhsCls = pinnedOfRhs(A.RHS)) {
      for (Expr *L : A.LHS) {
        if (auto *LN = dynamic_cast<NameExpr *>(L)) {
          if (LN->Ref && LN->Ref->Kind != BindingKind::Class)
            LN->Ref->PinnedClass = RhsCls;
        }
      }
    }
    break;
  }
  case NodeKind::IfStmt: {
    auto &I = static_cast<IfStmt &>(St);
    if (I.Cond) resolveExpr(*I.Cond, S);
    if (I.Then) resolveBlock(*I.Then, S);
    for (auto &EI : I.Elseifs) {
      if (EI.Cond) resolveExpr(*EI.Cond, S);
      if (EI.Body) resolveBlock(*EI.Body, S);
    }
    if (I.Else) resolveBlock(*I.Else, S);
    break;
  }
  case NodeKind::ForStmt: {
    auto &F = static_cast<ForStmt &>(St);
    if (F.Iter) resolveExpr(*F.Iter, S);
    if (F.Body) resolveBlock(*F.Body, S);
    break;
  }
  case NodeKind::WhileStmt: {
    auto &W = static_cast<WhileStmt &>(St);
    if (W.Cond) resolveExpr(*W.Cond, S);
    if (W.Body) resolveBlock(*W.Body, S);
    break;
  }
  case NodeKind::SwitchStmt: {
    auto &Sw = static_cast<SwitchStmt &>(St);
    if (Sw.Discriminant) resolveExpr(*Sw.Discriminant, S);
    for (auto &C : Sw.Cases) {
      if (C.Value) resolveExpr(*C.Value, S);
      if (C.Body)  resolveBlock(*C.Body, S);
    }
    break;
  }
  case NodeKind::TryStmt: {
    auto &T = static_cast<TryStmt &>(St);
    if (T.TryBody) resolveBlock(*T.TryBody, S);
    if (T.CatchBody) resolveBlock(*T.CatchBody, S);
    break;
  }
  case NodeKind::CommandStmt: {
    auto &C = static_cast<CommandStmt &>(St);
    Binding *B = S->lookup(C.Name);
    if (!B) {
      Diag.error(C.Range.Begin,
                 std::string("undefined command or function '") +
                     std::string(C.Name) + "'");
    }
    break;
  }
  default:
    break;
  }
}

void Resolver::resolveLValue(Expr &E, Scope *S) {
  switch (E.Kind) {
  case NodeKind::NameExpr: {
    auto &N = static_cast<NameExpr &>(E);
    Binding *B = S->lookup(N.Name);
    if (!B) {
      // Should have been pre-declared by collectAssignments; if not, emit.
      Diag.error(N.Range.Begin,
                 std::string("cannot assign to undeclared name '") +
                     std::string(N.Name) + "'");
      return;
    }
    if (B->Kind == BindingKind::Function || B->Kind == BindingKind::Builtin ||
        B->Kind == BindingKind::Class) {
      Diag.error(N.Range.Begin,
                 std::string("cannot assign to function '") +
                     std::string(N.Name) + "'");
    }
    N.Ref = B;
    B->WrittenTo = true;
    break;
  }
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<CallOrIndex &>(E);
    // `a(i) = x` — LHS must be indexing into a variable.
    resolveCallee(C, S);
    if (C.Resolved == CallKind::Call) {
      Diag.error(C.Range.Begin, "cannot assign to function call result");
    }
    for (Expr *A : C.Args) if (A) resolveExpr(*A, S);
    break;
  }
  case NodeKind::CellIndex: {
    auto &C = static_cast<CellIndex &>(E);
    if (C.Callee) resolveLValue(*C.Callee, S);
    for (Expr *A : C.Args) if (A) resolveExpr(*A, S);
    break;
  }
  case NodeKind::FieldAccess: {
    auto &F = static_cast<FieldAccess &>(E);
    if (F.Base) resolveLValue(*F.Base, S);
    break;
  }
  case NodeKind::DynamicField: {
    auto &F = static_cast<DynamicField &>(E);
    if (F.Base) resolveLValue(*F.Base, S);
    if (F.Name) resolveExpr(*F.Name, S);
    break;
  }
  default:
    Diag.error(E.Range.Begin, "expression is not assignable");
  }
}

void Resolver::resolveCallee(CallOrIndex &C, Scope *S) {
  // Resolve the callee first.
  if (C.Callee) resolveExpr(*C.Callee, S);

  // Decide Call vs Index.
  if (auto *N = dynamic_cast<NameExpr *>(C.Callee)) {
    if (N->Ref) {
      switch (N->Ref->Kind) {
      case BindingKind::Var:
      case BindingKind::Param:
      case BindingKind::Output:
      case BindingKind::Global:
      case BindingKind::Persistent:
        C.Resolved = CallKind::Index;
        N->Ref->ReadFrom = true;
        return;
      case BindingKind::Function:
      case BindingKind::Builtin:
      case BindingKind::Import:
      case BindingKind::Class:
        C.Resolved = CallKind::Call;
        return;
      }
    }
    // Unknown name with a callee that looks like an identifier — treat as
    // call and let type inference report it as ambiguous.
    C.Resolved = CallKind::Call;
    return;
  }

  /* Dot-method call: `obj.method(args)` parses as CallOrIndex whose
   * callee is a FieldAccess. Classify as Call so lowering emits a
   * method dispatch rather than a matlab.subscript when either:
   *   (a) the base variable is pinned to a class (instance method),
   *   (b) the base name itself resolves to a Class binding (static
   *       method), walking both chains up their Super ancestors. */
  if (auto *FA = dynamic_cast<FieldAccess *>(C.Callee)) {
    if (auto *BN = dynamic_cast<NameExpr *>(FA->Base)) {
      if (BN->Ref && BN->Ref->PinnedClass) {
        for (ClassDef *CC = BN->Ref->PinnedClass; CC; CC = CC->Super) {
          bool Found = false;
          for (matlab::Function *Mth : CC->Methods)
            if (Mth && Mth->Name == FA->Field) { Found = true; break; }
          if (Found) {
            C.Resolved = CallKind::Call;
            return;
          }
        }
      }
      if (BN->Ref && BN->Ref->Kind == BindingKind::Class &&
          BN->Ref->ClassDef) {
        for (ClassDef *CC = BN->Ref->ClassDef; CC; CC = CC->Super) {
          bool Found = false;
          for (matlab::Function *Mth : CC->StaticMethods)
            if (Mth && Mth->Name == FA->Field) { Found = true; break; }
          if (Found) {
            C.Resolved = CallKind::Call;
            return;
          }
        }
      }
    }
  }

  // Non-identifier callee: could be a function handle call, a chained index,
  // etc. Default to Call for handles, else Index.
  if (C.Callee && C.Callee->Ty &&
      C.Callee->Ty->K == Type::Kind::FuncHandle) {
    C.Resolved = CallKind::Call;
  } else {
    C.Resolved = CallKind::Index;
  }
}

void Resolver::resolveExpr(Expr &E, Scope *S) {
  switch (E.Kind) {
  case NodeKind::NameExpr: {
    auto &N = static_cast<NameExpr &>(E);
    Binding *B = S->lookup(N.Name);
    /* REPL workspace shadowing: when the same identifier exists as
     * BOTH a registered builtin (e.g. `grid` — a plotting function)
     * AND a workspace variable from a prior input, the user almost
     * certainly meant the variable (they assigned to it earlier).
     * In a single TU the resolver gets this right because the assign
     * gets ahead of any read — `grid = ...; grid(:)` resolves `grid`
     * locally on the read, shadowing the builtin.  In REPL the assign
     * lives in a previous TU's scope and only the workspace knows
     * about it, so the lookup falls back to the builtin and `grid(:)`
     * gets lowered as a function call.  Force the auto-declare path
     * when WorkspaceKindHook reports a real value for this name. */
    if (B && ReplMode && WorkspaceKindHook &&
        (B->Kind == BindingKind::Builtin ||
         B->Kind == BindingKind::Function) &&
        WorkspaceKindHook(N.Name.data(), (int64_t)N.Name.size()) >= 0) {
      B = nullptr;
    }
    if (!B) {
      /* REPL mode: names that don't resolve are auto-declared as Vars
       * in the current (script) scope. Lowering then routes the read
       * through the runtime workspace, which holds values produced by
       * earlier REPL inputs. This is the right trade-off for an
       * interactive session where each input is its own TU but the
       * user expects identifiers to persist. */
      if (ReplMode) {
        Binding *NB = Sema.newBinding();
        B = S->getOrDeclareVar(N.Name, NB);
        /* REPL cross-input persistence: when the resolver auto-declares
         * a name that wasn't assigned in this TU, query the live
         * workspace for the kind under which a prior input bound it.
         * Stamping InferredType lets every downstream dispatch site
         * (string disp, strlen, isstring, the workspace load path
         * itself) see the correct shape — without this, a fresh
         * `disp(t)` after an earlier `t = "..."` couldn't tell
         * matrix from string and either silently dropped the read or
         * matrix-cast the descriptor's bytes. */
        if (WorkspaceKindHook) {
          int K = WorkspaceKindHook(N.Name.data(), (int64_t)N.Name.size());
          /* Kind encoding (matches matlab_dbg_ws_kind):
           *   0 = f64 scalar, 1 = matlab_mat*, 2 = matlab_obj*,
           *   3 = matlab_string*, 4 = matlab_mat_u8*, 5 = matlab_mat_i32*.
           *   Stamp InferredType for the kinds the lowering can
           *   specialise on — without a scalar-double stamp, the
           *   workspace load path falls back to matlab_ws_get_mat
           *   (returns ptr) and downstream consumers that need an f64
           *   (num2str / arithmetic / range bounds) silently misbehave.
           *   Kinds 4/5 (typed-int matrices) are stamped with a non-
           *   scalar Array(UInt8/Int32, unknown) so the BinaryOp emit
           *   site picks the typed runtime entry points. */
          if (K == 0) NB->InferredType = TC.scalar(Dtype::Double);
          /* Kind 1 = generic matlab_mat* (real double matrix). Without
           * this stamp, downstream MLIR lowerings fall back to a ptr-
           * typed workspace load and elementwise operations (`.*`,
           * matrix `+ -`, `norm`, etc.) misdispatch on the next REPL
           * turn. Stamp the canonical "double array of unknown shape"
           * so the BinaryOp / call_builtin path picks the matrix-mat
           * runtime entries. */
          else if (K == 1) NB->InferredType = TC.arrayOf(Dtype::Double, Shape::unknown());
          else if (K == 3) NB->InferredType = TC.stringScalar();
          else if (K == 4) NB->InferredType = TC.arrayOf(Dtype::UInt8, Shape::unknown());
          else if (K == 5) NB->InferredType = TC.arrayOf(Dtype::Int32, Shape::unknown());
          /* Kind 7 = matlab_sym* (Phase 6 — Symbolic Math Toolbox).
           * No InferredType is set (Sema has no SymType yet), but the
           * binding is tagged so the MLIR lowering's BinaryOp / disp /
           * workspace-store dispatch sees it as sym across TUs. */
          else if (K == 7) NB->IsSym = true;
          /* Kind 8 = matlab_symmat* (Phase 6.1 — symbolic matrix). */
          else if (K == 8) NB->IsSymmat = true;
          /* Kind 12 = matlab_struct* (plain field-holder, not a
           * classdef instance).  Stamping IsStruct lets the MLIR
           * lowering re-populate StructInitialised on the next
           * compile so `lb.PathLoss` and friends route through the
           * struct-get path. */
          else if (K == 12) NB->IsStruct = true;
          /* Kind 2 = matlab_obj* (classdef instance).  Re-pin the
           * binding to the runtime-tracked class so the obj-call
           * sugar / dot-method dispatch / class operator overloads
           * stay live across REPL turns.  Without this, a cross-
           * input `crc(1)` (when `crc` was a SystemObject set in a
           * prior turn) would fall through to the matrix-subscript
           * path and read garbage.  The classdef itself must be in
           * scope by this point — the REPL prelude loader pulls it
           * in via the same workspace-scan path. */
          else if (K == 2 && WorkspaceClassNameHook) {
            int64_t CnLen = 0;
            const char *Cn = WorkspaceClassNameHook(
                N.Name.data(), (int64_t)N.Name.size(), &CnLen);
            if (Cn && CnLen > 0) {
              std::string_view CnView(Cn, (size_t)CnLen);
              Binding *CB = S->lookup(CnView);
              if (CB && CB->Kind == BindingKind::Class && CB->ClassDef)
                NB->PinnedClass = CB->ClassDef;
            }
          }
        }
      } else {
        Diag.error(N.Range.Begin,
                   std::string("undefined name '") + std::string(N.Name) + "'");
        return;
      }
    }
    N.Ref = B;
    B->ReadFrom = true;
    break;
  }
  case NodeKind::BinaryOp: {
    auto &B = static_cast<BinaryOpExpr &>(E);
    if (B.LHS) resolveExpr(*B.LHS, S);
    if (B.RHS) resolveExpr(*B.RHS, S);
    break;
  }
  case NodeKind::UnaryOp: {
    auto &U = static_cast<UnaryOpExpr &>(E);
    if (U.Operand) resolveExpr(*U.Operand, S);
    break;
  }
  case NodeKind::PostfixOp: {
    auto &P = static_cast<PostfixOpExpr &>(E);
    if (P.Operand) resolveExpr(*P.Operand, S);
    break;
  }
  case NodeKind::RangeExpr: {
    auto &R = static_cast<RangeExpr &>(E);
    if (R.Start) resolveExpr(*R.Start, S);
    if (R.Step)  resolveExpr(*R.Step, S);
    if (R.End)   resolveExpr(*R.End, S);
    break;
  }
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<CallOrIndex &>(E);
    resolveCallee(C, S);
    for (Expr *A : C.Args) if (A) resolveExpr(*A, S);
    break;
  }
  case NodeKind::CellIndex: {
    auto &C = static_cast<CellIndex &>(E);
    if (C.Callee) resolveExpr(*C.Callee, S);
    for (Expr *A : C.Args) if (A) resolveExpr(*A, S);
    break;
  }
  case NodeKind::FieldAccess: {
    auto &F = static_cast<FieldAccess &>(E);
    if (F.Base) resolveExpr(*F.Base, S);
    break;
  }
  case NodeKind::DynamicField: {
    auto &F = static_cast<DynamicField &>(E);
    if (F.Base) resolveExpr(*F.Base, S);
    if (F.Name) resolveExpr(*F.Name, S);
    break;
  }
  case NodeKind::MatrixLiteral: {
    auto &M = static_cast<MatrixLiteral &>(E);
    for (auto &R : M.Rows)
      for (Expr *C : R) if (C) resolveExpr(*C, S);
    break;
  }
  case NodeKind::CellLiteral: {
    auto &M = static_cast<CellLiteral &>(E);
    for (auto &R : M.Rows)
      for (Expr *C : R) if (C) resolveExpr(*C, S);
    break;
  }
  case NodeKind::AnonFunction: {
    auto &A = static_cast<AnonFunction &>(E);
    Scope *Inner = Sema.newScope(S, "<anon>");
    A.ParamRefs.clear();
    for (auto P : A.Params) {
      Binding *B = Sema.newBinding();
      Binding *D = Inner->declare(P, BindingKind::Param, B);
      A.ParamRefs.push_back(D);
    }
    if (A.Body) resolveExpr(*A.Body, Inner);
    break;
  }
  case NodeKind::FuncHandle: {
    auto &F = static_cast<FuncHandle &>(E);
    F.Ref = S->lookup(F.Name);
    if (!F.Ref) {
      Diag.error(F.Range.Begin,
                 std::string("undefined function '") + std::string(F.Name) +
                     "' in function handle");
    } else if (F.Ref->Kind != BindingKind::Function &&
               F.Ref->Kind != BindingKind::Builtin) {
      Diag.error(F.Range.Begin,
                 std::string("'") + std::string(F.Name) +
                     "' is not a function");
    }
    break;
  }
  // Literals and EndExpr/ColonExpr need no resolution.
  default:
    break;
  }
}

} // namespace matlab
