#include "matlab/Sema/Resolver.h"
#include "matlab/Sema/Type.h"

#include <algorithm>
#include <functional>
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
    /* Degree-argument trigonometry. */
    "sind", "cosd", "tand", "asind", "acosd", "atand", "atan2d",
    "log2", "log10", "sign",
    "min", "max", "sum", "prod", "mean",
    "cumsum", "cumprod",
    "sort", "sortrows", "unique", "ismember",
    "setdiff", "intersect", "union",
    "horzcat", "vertcat", "permute", "ipermute", "squeeze", "cat",
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
    /* Phase 5.4 — timetable. Constructor + table promotion; the
     * subscripting / retime / synchronize entries live in the same
     * timetable namespace and ride the same builtin registration. */
    "timetable", "istimetable", "table2timetable", "timetable2table",
    "timerange", "retime", "synchronize", "withtol",
    "fillmissing", "summary", "head", "tail", "rowfun",
    /* Phase 5.4 Financial indicators on timetables. */
    "movavg", "macd", "sma", "ema", "highlow", "candle",
    /* Financial Toolbox Tier-1: date arithmetic. */
    "yearfrac", "daysdif", "daysadd", "daysact", "days360", "days365",
    "busdate", "isbusday", "eomdate", "lweekdate", "fweekdate",
    "m2xdate", "x2mdate",
    /* Tier-1: cash flows + depreciation. */
    "pvfix", "fvfix", "pvvar", "fvvar", "irr", "payper", "amortize",
    "nomrr", "effrr", "depstln", "depsoyd", "depfixdb",
    /* Tier-1: bond pricing. */
    "bndprice", "bndyield", "bnddurp", "bnddury", "bndconvp",
    "bndconvy", "accrfrac", "cfdates", "cfamounts",
    "prdisc", "prtbill", "ytbill", "beytbill",
    /* Tier-1: returns + indicator function-form. */
    "tick2ret", "ret2tick", "rsindex", "bolling",
    /* Tier-2: performance metrics + Black-Scholes. */
    "sharpe", "sortino", "inforatio", "tracking", "maxdrawdown",
    "lpm", "elpm", "portalpha",
    "blsprice", "blsdelta", "blsgamma", "blsvega", "blsrho", "blstheta",
    "blslambda", "blsimpv",
    /* Tier-3: Portfolio classdef methods. The class itself
     * (`Portfolio`) is provided by finance_classdefs.m via the
     * auto-prepend mechanism; the methods listed here become first-
     * arg-class-pinned dispatches through the LowerTensorOps Spec
     * table when called as `setAssetMoments(p, m, C)`. */
    "setAssetMoments", "setBounds", "setBudget",
    "setEquality", "setInequality", "setDefaultConstraints",
    "estimateFrontier", "estimateFrontierByReturn", "estimateFrontierByRisk",
    "estimatePortMoments", "estimatePortReturn", "estimatePortRisk",
    "estimateMaxSharpeRatio", "estimateAssetMoments", "plotFrontier",
    "estimateBounds", "estimatePortFrontier", "blacklitterman",
    "riskparity", "riskbudget", "riskcontribution",
    /* Tier-4: regression with missing data + credit risk. */
    "ecmnmle", "ecmncov", "mvnrmle", "capm",
    "transprob", "cdsbootstrap", "cdsspread", "cdsprice",
    "creditscorecard", "autobinning", "bininfo", "fitmodel",
    "score", "probdefault",
    /* Tier-5: CVaR / MAD portfolio methods (shared names route on
     * the RiskKind discriminant inside the runtime). */
    "PortfolioCVaR", "PortfolioMAD", "setScenarios",
    "setProbabilityLevel", "estimatePortVaR",
    "backtest", "backtestSummary",
    /* Tier-6: SDE Monte Carlo. */
    "gbm", "bm", "cir", "hwv", "simByEuler", "simBySolution",
    "haltonseq", "optpricemc",
    /* ----------------------------------------------------------------
     * Econometrics Toolbox (docs/econometrics_toolbox_roadmap.md).
     * Tier-1 is entirely function-form and routes through the
     * LowerTensorOps Spec table.  Model-object method names that
     * collide with System-Identification (estimate/forecast/infer/
     * simulate/filter) are dispatched class-pinned in Lowering, so they
     * coexist.
     * ---------------------------------------------------------------- */
    /* Tier-1: data prep. */
    "price2ret", "ret2price", "hpfilter",
    /* Tier-1: ACF / PACF. */
    "autocorr", "parcorr", "crosscorr",
    /* Tier-1: diagnostic + comparison tests. */
    "lbqtest", "archtest", "aicbic", "lmtest", "waldtest", "lratiotest",
    "hac", "fgls",
    /* Tier-1: unit-root + stationarity tests. */
    "adftest", "pptest", "kpsstest", "lmctest", "vratiotest",
    /* Tier-2+: model-object methods (class-pinned in Lowering on the
     * receiver class so they coexist with System-Identification's
     * idpoly forecast/etc).  `forecast`/`filter` are already builtins.
     * The model constructors (arima/garch/egarch/gjr/...) are NOT
     * builtins — they resolve via the classdef auto-prepend. */
    "estimate", "infer", "simulate", "summarize",
    /* Tier-4: VAR impulse-response method + cointegration tests. */
    "irf", "egcitest", "jcitest", "jcontest",
    /* Tier-5: state-space methods (filter is already a builtin). */
    "smooth",
    /* Tier-6: Markov-chain method (asymptotics). */
    "asymptotics",
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
    /* Tier 2.2 — continuous->discrete state-space ZOH + Tustin, and the
     * ZOH discrete->continuous inverse d2c. */
    "c2d", "c2d_tustin", "d2c",
    /* Model-data extractors — registered as builtins so function-style
     * class-method dispatch routes `ssdata(sys)` / `tfdata(G)` to the
     * ss / tf classdef methods (function-form dispatch only fires for a
     * known-builtin callee, see the dispatch in resolveExpr). */
    "ssdata", "tfdata",
    /* margin(sys) — multi-return gain/phase-margin splitter handled in the
     * lowering's model-object multi-return path. */
    "margin",
    /* MPC Toolbox Tier-1/2/3 — runtime-symbol builtins called from
     * inside `mpc_classdefs.m`.  5 entries: construct (mutates the
     * mpc obj at construction time), move (single tick), move_opt
     * (Tier-2 5-arg with run-time overrides), move_adaptive (Tier-3
     * 7-arg with per-tick plant update), sim (closed-loop sim). */
    "matlab_mpc_construct", "matlab_mpc_move", "matlab_mpc_move_opt",
    "matlab_mpc_move_adaptive", "matlab_mpc_move_tv", "matlab_mpc_sim",
    /* Global Optimization Toolbox Tier-1 — stochastic global solvers.
     * User-facing names + the runtime symbols (the Lowering dispatch
     * remaps the MATLAB call forms onto the 5-arg runtime entries). */
    "ga", "particleswarm", "simulannealbnd",
    "matlab_gads_ga", "matlab_gads_particleswarm",
    "matlab_gads_simulannealbnd",
    /* Global Optimization Toolbox Tier-2 — multi-start meta-solvers. */
    "createOptimProblem", "run",
    "matlab_gads_make_problem", "matlab_gads_multistart",
    "matlab_gads_globalsearch",
    /* Global Optimization Toolbox Tier-3 — direct search. */
    "patternsearch", "matlab_gads_patternsearch",
    /* Global Optimization Toolbox Tier-4 — surrogate optimization. */
    "surrogateopt", "matlab_gads_surrogateopt",
    /* Global Optimization Toolbox Tier-5 — multiobjective. */
    "gamultiobj", "paretosearch",
    "matlab_gads_gamultiobj", "matlab_gads_paretosearch",
    /* Global Optimization Toolbox Tier-6 — options + integer ga. */
    "optimoptions", "matlab_gads_ga_opts",
    /* Tier-2 run(solver, problem) — runtime-dispatched (REPL-safe). */
    "matlab_gads_run",
    /* Statistics and Machine Learning Toolbox Tier-1 — descriptive stats,
     * covariance/correlation, probability distributions, and RNGs.  These
     * are user-facing names; pde_table rewrites each to its matlab_stats_*
     * symbol (boxing scalar args). */
    "prctile", "quantile", "iqr", "range", "mode", "skewness", "kurtosis",
    "geomean", "harmmean", "cov", "corr", "corrcoef",
    "normpdf", "normcdf", "norminv", "exppdf", "expcdf", "expinv",
    "unifpdf", "unifcdf", "unifinv",
    "normrnd", "unifrnd", "exprnd",
    /* Distribution objects (makedist/fitdist + pdf/cdf/icdf/random methods,
     * runtime-dispatched on the object's class). */
    "makedist", "fitdist", "pdf", "cdf", "icdf", "random",
    "matlab_stats_fitdist_init", "matlab_stats_pd_pdf", "matlab_stats_pd_cdf",
    "matlab_stats_pd_icdf", "matlab_stats_pd_random",
    /* Tier-2 — hypothesis tests + ANOVA. */
    "ttest", "ttest2", "vartest2", "ztest", "kstest",
    "ranksum", "signrank", "signtest", "anova1",
    /* Tier-3 — regression. */
    "fitlm", "fitglm", "ridge", "regress",
    "matlab_stats_fitlm_init", "matlab_stats_fitglm_init",
    "matlab_stats_lm_predict", "matlab_stats_ridge", "matlab_stats_regress",
    /* Tier-4 — PCA + clustering. */
    "pca", "kmeans", "pdist2", "pdist", "squareform", "silhouette",
    /* Tier-5 — classification. */
    "fitcknn", "fitcnb", "fitcdiscr", "fitctree", "fitcsvm", "fitcecoc",
    "confusionmat",
    /* Tier-6 — ensembles. */
    "fitcensemble", "TreeBagger", "matlab_stats_fitensemble_init",
    /* Tier-6 — Hidden Markov Models. */
    "hmmgenerate", "hmmviterbi", "hmmdecode", "hmmtrain",
    "matlab_stats_hmmgenerate", "matlab_stats_hmmviterbi",
    "matlab_stats_hmmdecode", "matlab_stats_hmmtrain",
    "matlab_stats_hmm_states", "matlab_stats_hmm_logp", "matlab_stats_hmm_emis",
    /* Tier-6 — Bayesian optimization. */
    "bayesopt", "matlab_stats_bayesopt",
    /* ===== Image Processing Toolbox Tier-1 + Tier-2 ===== */
    "imread", "imwrite", "checkerboard",
    "im2double", "im2single", "im2uint8", "rgb2gray", "im2gray", "mat2gray",
    "imadd", "imsubtract", "immultiply", "imdivide", "imabsdiff",
    "imcomplement", "imlincomb",
    "imhist", "imadjust", "stretchlim", "mean2", "std2",
    "fspecial", "imgaussfilt", "imboxfilt", "medfilt2", "ordfilt2",
    "stdfilt", "rangefilt",
    "histeq", "adapthisteq", "imsharpen", "imhistmatch", "imnoise",
    /* Image Processing Tier-3 — geometric transforms. */
    "imresize", "imrotate", "imcrop", "imtranslate", "imwarp", "fitgeotform2d",
    "matlab_image_fitgeo_init",
    /* Image Processing Tier-4 — binarization + morphology. */
    "graythresh", "otsuthresh", "imbinarize", "im2bw", "strel",
    "imerode", "imdilate", "imopen", "imclose", "imtophat", "imbothat",
    "imfill", "edge", "bwareaopen",
    /* Image Processing Tier-5 — segmentation + region analysis. */
    "bwlabel", "regionprops", "label2rgb", "bweuler", "imsegkmeans",
    /* Image Processing Tier-6 — transforms/quality/ROI/colour/block/deblur. */
    "immse", "psnr", "ssim",
    "rgb2hsv", "hsv2rgb", "rgb2ycbcr", "ycbcr2rgb", "rgb2lab", "lab2rgb",
    "dct2", "idct2", "radon", "hough", "houghpeaks",
    "poly2mask", "roifilt2", "im2col", "col2im",
    "deconvwnr", "edgetaper",
    "matlab_stats_fitknn_init", "matlab_stats_fitnb_init", "matlab_stats_fitlda_init",
    "matlab_stats_fittree_init", "matlab_stats_fitsvm_init", "matlab_stats_fitecoc_init",
    "matlab_stats_clf_predict", "matlab_stats_confusionmat",
    /* ===== Curve Fitting Toolbox Tier-1 + Tier-2 + Tier-3 ===== */
    "fit", "feval", "coeffvalues",
    "differentiate", "integrate", "confint", "formula", "numcoeffs",
    "matlab_curvefit_fit", "matlab_curvefit_fit_opts", "matlab_curvefit_feval",
    "matlab_curvefit_coeffvalues", "matlab_curvefit_gof",
    "matlab_curvefit_output", "matlab_curvefit_disp",
    "matlab_curvefit_fittype_init", "matlab_curvefit_fit_custom",
    "matlab_curvefit_confint", "matlab_curvefit_differentiate",
    "matlab_curvefit_integrate", "matlab_curvefit_numcoeffs",
    "matlab_curvefit_formula",
    /* Tier-4 — smoothing + smoothing spline. */
    "smooth", "csaps",
    "matlab_curvefit_smooth1", "matlab_curvefit_smooth2", "matlab_curvefit_smooth3",
    "matlab_curvefit_csaps",
    /* Tier-5 — surface fitting. */
    "matlab_curvefit_fit_surface", "matlab_curvefit_sfeval",
    /* Tier-6 — ppform spline layer. */
    "spline", "pchip", "ppmak", "fnval", "fnder", "fnint", "fnbrk",
    "matlab_curvefit_spline_init", "matlab_curvefit_ppmak_init",
    "matlab_curvefit_fnval", "matlab_curvefit_fnder_init",
    "matlab_curvefit_fnint_init", "matlab_curvefit_fnbrk",
    /* ===== Wavelet Toolbox ===== *
     * Only the USER-facing names are registered.  Every mapping to a
     * runtime symbol goes through the pde_table (keyed by these names,
     * which also performs the const_char→matlab_string coercion) or the
     * wavelet multi-return dispatch — registering a `matlab_wavelet_*`
     * alias here would hijack the user name before the pde_table runs
     * (the curve-fit `smooth` precedent). */
    /* Tier-1 */
    "dwt", "idwt", "wavedec", "waverec", "wfilters",
    "appcoef", "detcoef", "wrcoef", "upcoef", "upwlev",
    "wextend", "wkeep", "wmaxlev", "dwtmode", "qmf",
    "wentropy", "wenergy", "centfrq", "wavefun", "waveinfo",
    /* Tier-2 */
    "wthresh", "thselect", "wnoisest", "wnoise", "ddencmp",
    "wden", "wdencmp", "wdenoise", "wcompress", "measerr", "wpdencmp",
    /* Tier-3 */
    "cwt", "icwt", "scal2frq", "freq2scal", "wcoherence",
    /* Tier-4 */
    "swt", "iswt", "modwt", "imodwt", "modwtmra", "modwtvar",
    "dwt2", "idwt2", "wavedec2", "waverec2", "appcoef2", "detcoef2",
    "wrcoef2", "wcodemat",
    /* Tier-5/6 */
    "wpdec", "wprec", "wpcoef", "wprcoef", "besttree", "bestlevt",
    "ewt", "vmd", "emd", "tqwt", "itqwt", "matchingPursuit",
    "waveletScattering", "featureMatrix",
    /* ===== DSP System Toolbox ===== *
     * The `dsp.*` System Objects live in dsp_classdefs.m; the user-facing
     * surface is the constructor + `obj(frame)` call-syntax (lowered to
     * step) — no bare builtin names here.  These `matlab_dsp_*` runtime
     * symbols are the entries the classdef `step`/`reset` method bodies
     * forward `obj` into (wired through the pde_table). */
    /* Tier-1 — filter-object step + lifecycle. */
    "matlab_dsp_iir_step", "matlab_dsp_sos_step", "matlab_dsp_delay_step",
    "matlab_dsp_reset", "matlab_dsp_init_state", "matlab_dsp_get_state",
    /* Tier-2 — filter design (function-form, user-facing). */
    "firpm", "firpmord", "firls", "kaiserord",
    "iirnotch", "iirpeak", "iircomb", "designfilt",
    /* Tier-3 — adaptive-filter step + lifecycle. */
    "matlab_dsp_lms_step", "matlab_dsp_rls_step", "matlab_dsp_get_weights",
    /* Tier-4 — multirate step + filter-bank helpers. */
    "matlab_dsp_firdecim_step", "matlab_dsp_firinterp_step",
    "matlab_dsp_cicdecim_step", "matlab_dsp_cicinterp_step",
    "matlab_dsp_rateconv_step", "matlab_dsp_channelizer_step",
    "matlab_dsp_synthesizer_step",
    /* Tier-5 — sources, sliding stats, detectors, spectral, buffering. */
    "matlab_dsp_sine_step", "matlab_dsp_nco_step", "matlab_dsp_chirp_step",
    "matlab_dsp_movavg_step", "matlab_dsp_movrms_step",
    "matlab_dsp_movmax_step", "matlab_dsp_movmin_step", "matlab_dsp_movstd_step",
    "matlab_dsp_peakfind_step", "matlab_dsp_dcblock_step",
    "matlab_dsp_zcd_step", "matlab_dsp_spectest_step",
    "matlab_dsp_asyncbuf_write", "matlab_dsp_asyncbuf_read",
    /* Tier-5 user-facing buffering. */
    "buffer",
    /* Tier-6 — linalg SOs + polish filter SOs. */
    "matlab_dsp_levinson_step", "matlab_dsp_notchpeak_step",
    "matlab_dsp_lowpass_step",  "matlab_dsp_highpass_step",
    /* Tier-7 / Tier-8 — dsphdl.* simulation step entries + CORDIC math. */
    "matlab_dsphdl_fir_step", "matlab_dsphdl_biquad_step",
    "matlab_dsphdl_sine_step", "matlab_dsphdl_nco_step",
    "matlab_dsphdl_firdecim_step", "matlab_dsphdl_cicdecim_step",
    "matlab_dsphdl_latency",
    "matlab_dsp_cordic_atan2", "matlab_dsp_cordic_sqrt",
    /* T8 user-facing CORDIC. */
    "cordic_atan2", "cordic_sqrt",
    /* MPC Tier-4 §5.4 — standalone active-set QP solver. */
    "matlab_mpc_active_set",
    /* MPC Tier-4 §5.1/5.2/5.3 — explicit MPC. */
    "matlab_mpc_generate_explicit", "matlab_mpc_move_explicit",
    "matlab_mpc_simplify_explicit",
    /* MPC Tier-4 §5.7 — Finite Control Set MPC. */
    "matlab_mpc_move_finite",
    /* MPC Tier-5 — Nonlinear MPC. */
    "matlab_nlmpc_move",
    /* User-facing class-method names — registered as builtins so
     * call-site `mpcmove(obj, ...)` / `sim(obj, ...)` / etc. route
     * through class-method dispatch when `obj` is mpc-pinned. */
    "mpcmove", "sim", "mpcmoveAdaptive", "mpcmoveTV",
    "mpcActiveSetSolver",
    "generateExplicitMPC", "mpcmoveExplicit", "mpcmoveFinite",
    "nlmpcmove",
    /* MPC Tier-6 §7.4 — estimator accessors. */
    "setEstimator", "getEstimator",
    /* MPC Tier-6 §7.5 — review() diagnostic. */
    "review", "matlab_mpc_review",
    /* MPC Tier-6 §7.6 — sim() with mpcsimopt. */
    "matlab_mpc_sim_opt",
    /* System Identification Toolbox Tier-1 — user-facing estimator /
     * validation / metric names.  Dispatched in Lowering.cpp's
     * class-pinned-first-arg path to the matlab_ident_* runtime
     * entries when arg0 is iddata / idpoly pinned.  (`sim` is already
     * registered above for MPC; the dispatch keys on the pinned class
     * so the two coexist.)  The matlab_ident_* runtime symbols are
     * registered too for the loose-match table. */
    "arx", "ar", "predict", "compare", "goodnessOfFit", "fpe", "aic",
    "matlab_ident_arx", "matlab_ident_ar", "matlab_ident_sim",
    "matlab_ident_predict", "matlab_ident_compare",
    "matlab_ident_goodness", "matlab_ident_fpe", "matlab_ident_aic",
    "matlab_ident_poly2ss_A", "matlab_ident_poly2ss_B",
    "matlab_ident_poly2ss_C", "matlab_ident_poly2ss_D",
    /* Tier-2 — PEM estimators + residual analysis + IV + delay. */
    "armax", "oe", "bj", "pe", "resid", "iv4", "delayest",
    "matlab_ident_armax", "matlab_ident_oe", "matlab_ident_bj",
    "matlab_ident_pe", "matlab_ident_resid",
    "matlab_ident_iv4", "matlab_ident_delayest",
    /* Tier-3 — subspace state-space + TF estimation. */
    "n4sid", "ssest", "tfest",
    "matlab_ident_n4sid", "matlab_ident_ssest", "matlab_ident_tfest",
    "matlab_ident_sim_ss", "matlab_ident_compare_ss",
    "matlab_ident_ss_A", "matlab_ident_ss_B",
    "matlab_ident_ss_C", "matlab_ident_ss_D",
    /* Tier-4 — grey-box + impulse response + forecast + spectral. */
    "greyest", "matlab_ident_greyest",
    "impulseest", "forecast",
    "matlab_ident_impulseest", "matlab_ident_forecast",
    "etfe", "spa", "matlab_ident_etfe", "matlab_ident_spa",
    /* Tier-5 — nonlinear state estimation (EKF/UKF) + recursive RLS. */
    "correct", "matlab_ident_ekf_init",
    "matlab_ident_ekf_predict", "matlab_ident_ekf_correct",
    "matlab_ident_ukf_predict", "matlab_ident_ukf_correct",
    "matlab_ident_rls_init", "matlab_ident_rls_step",
    "matlab_ident_rarx_init", "matlab_ident_rarx_step",
    "nlgreyest", "matlab_ident_nlgreyest",
    /* Tier-6 — regularized arx + parameter introspection. */
    "getcov", "getpvec", "setpvec",
    "matlab_ident_arx_reg", "matlab_ident_getcov",
    "matlab_ident_getpvec", "matlab_ident_setpvec",
    /* ===== Sensor Fusion and Tracking Toolbox Tier-1 — quaternion + rotations ====
     * Class names (quaternion / trackingKF / trackingEKF / trackingUKF /
     * objectDetection / imuSensor / gpsSensor / ahrsfilter / imufilter /
     * complementaryFilter / insfilterMARG) are NOT registered here — they're
     * picked up from fusion_classdefs.m by the prelude loader and resolved
     * as classes, so calls route to Lowering's constructor-intercept path. */
    "ecompass", "slerp",
    "rotatepoint", "rotateframe",
    "quat2eul", "eul2quat", "quat2rotm", "rotm2quat",
    "matlab_fusion_quat_init_wxyz", "matlab_fusion_quat_init_mat",
    "matlab_fusion_quat_init_from_data",
    "matlab_fusion_quat_mul_data", "matlab_fusion_quat_conj_data",
    "matlab_fusion_quat_norm_data", "matlab_fusion_quat_normalize_data",
    "matlab_fusion_quat_inverse_data",
    "matlab_fusion_quat_to_rotm", "matlab_fusion_quat2rotm",
    "matlab_fusion_rotm_to_quat",
    "matlab_fusion_quat_to_eul", "matlab_fusion_eul_to_quat",
    "matlab_fusion_quat_rotatepoint", "matlab_fusion_quat_rotateframe",
    "matlab_fusion_quat_slerp", "matlab_fusion_quat_dist",
    "matlab_fusion_ecompass",
    "matlab_fusion_quat_disp", "matlab_fusion_quat_parts",
    /* ===== Sensor Fusion Tier-1 small core gaps (also generic builtins) ===== */
    "cross", "dot", "deg2rad", "rad2deg", "mvnrnd",
    "matlab_cross", "matlab_dot", "matlab_deg2rad", "matlab_rad2deg",
    "matlab_normalize_vec", "matlab_mvnrnd",
    /* ===== Sensor Fusion Tier-2 — tracking filters + motion/measurement models ===== */
    "constvel", "constacc", "constturn",
    "constveljac", "constaccjac", "constturnjac",
    "cvmeas", "cameas", "ctmeas",
    "cvmeasjac", "cameasjac", "ctmeasjac",
    "initcvekf", "initctekf", "initcakf",
    "matlab_fusion_trackingkf_init", "matlab_fusion_trackingkf_predict",
    "matlab_fusion_trackingkf_correct", "matlab_fusion_trackingkf_distance",
    "matlab_fusion_trackingkf_likelihood",
    "matlab_fusion_trackingekf_init", "matlab_fusion_trackingekf_predict",
    "matlab_fusion_trackingekf_correct",
    "matlab_fusion_trackingukf_init", "matlab_fusion_trackingukf_predict",
    "matlab_fusion_trackingukf_correct",
    "matlab_fusion_constvel", "matlab_fusion_constacc", "matlab_fusion_constturn",
    "matlab_fusion_cvmeas", "matlab_fusion_cameas", "matlab_fusion_ctmeas",
    "matlab_fusion_objdet_init",
    "matlab_fusion_initcvekf", "matlab_fusion_initctekf",
    /* ===== Sensor Fusion Tier-3 — inertial sensors + orientation fusion ===== */
    "allanvar",
    /* MARG fusion update entry names — `predict` already registered.  */
    "fuseaccel", "fusegps", "fusemag",
    /* ===== Sensor Fusion Tier-4 — trajectory + scenario =================== */
    "lookupPose", "lla2ned", "ned2lla",
    "matlab_fusion_waypoint_init", "matlab_fusion_waypoint_lookup",
    "matlab_fusion_lla2ned", "matlab_fusion_ned2lla",
    /* ===== Sensor Fusion Tier-5 — multi-object trackers + assignment ====== */
    "assignmunkres", "numConfirmed",
    "matlab_fusion_assignmunkres",
    "matlab_fusion_gnn_init", "matlab_fusion_gnn_step",
    "matlab_fusion_gnn_numconfirmed",
    /* ===== Sensor Fusion Tier-6 — track fusion + metrics + smoother ======= */
    "trackFuser", "trackGOSPAMetric", "trackOSPAMetric",
    "trackErrorMetrics", "rtsSmoother",
    "matlab_fusion_covint", "matlab_fusion_gospa", "matlab_fusion_ospa",
    "matlab_fusion_trackerror", "matlab_fusion_rts_smoother",
    /* ===== Robotics System Toolbox — T1 transforms + tform conversions ====
     * Class names (se3/so3/rigidBodyTree/inverseKinematics/...) are NOT
     * builtins — they resolve via the prelude.  Only the free functions
     * (conversions / utilities / IK helpers / planners) register here. */
    "trvec2tform", "tform2trvec", "rotm2tform", "tform2rotm",
    "eul2tform", "tform2eul", "axang2rotm", "rotm2axang",
    "axang2tform", "tform2axang", "quat2tform", "tform2quat",
    "homtrans", "wrapToPi", "wrapTo2Pi", "vecnorm",
    "matlab_robotics_se3_init", "matlab_robotics_so3_init",
    "matlab_robotics_trvec2tform", "matlab_robotics_tform2trvec",
    "matlab_robotics_rotm2tform", "matlab_robotics_tform2rotm",
    "matlab_robotics_eul2rotm", "matlab_robotics_rotm2eul",
    "matlab_robotics_eul2tform", "matlab_robotics_tform2eul",
    "matlab_robotics_axang2rotm", "matlab_robotics_rotm2axang",
    "matlab_robotics_axang2tform", "matlab_robotics_tform2axang",
    "matlab_robotics_quat2tform", "matlab_robotics_tform2quat",
    "matlab_robotics_homtrans",
    "matlab_robotics_wrapToPi", "matlab_robotics_wrapTo2Pi",
    "matlab_robotics_vecnorm",
    "matlab_robotics_tform_mul", "matlab_robotics_tform_inv",
    /* T2-T6 user names + runtime symbols.  All free functions; class
     * constructors land in Lowering's constructor-intercept path. */
    "addBody", "getTransform", "geometricJacobian",
    "homeConfiguration", "randomConfiguration", "loadrobot",
    "cubicpolytraj", "trapveltraj", "transformtraj",
    "massMatrix", "inverseDynamics",
    "forwardDynamics", "gravityTorque", "velocityProduct", "centerOfMass",
    "matlab_robotics_forwardDynamics", "matlab_robotics_gravityTorque",
    "matlab_robotics_velocityProduct", "matlab_robotics_centerOfMass",
    "importrobot", "matlab_robotics_importrobot",
    "generalizedInverseKinematics", "constraintPositionTarget",
    "constraintOrientationTarget", "constraintJointBounds",
    "matlab_robotics_gik_init", "matlab_robotics_gik_solve",
    "matlab_robotics_constraint_position_init",
    "matlab_robotics_constraint_orientation_init",
    "collisionCylinder", "collisionCapsule",
    "matlab_robotics_collcyl_init", "matlab_robotics_collcap_init",
    "matlab_robotics_gjk_collision",
    "setOccupancy", "getOccupancy", "checkOccupancy", "findpath",
    "checkCollision", "plan", "derivative",
    "matlab_robotics_tree_init", "matlab_robotics_tree_addbody",
    "matlab_robotics_loadrobot",
    "matlab_robotics_getTransform", "matlab_robotics_geometricJacobian",
    "matlab_robotics_homeConfiguration", "matlab_robotics_randomConfiguration",
    "matlab_robotics_ik_init", "matlab_robotics_ik_solve",
    "matlab_robotics_constraint_pose_init",
    "matlab_robotics_cubicpolytraj", "matlab_robotics_trapveltraj",
    "matlab_robotics_transformtraj",
    "matlab_robotics_massMatrix", "matlab_robotics_inverseDynamics",
    "matlab_robotics_diffdrive_init", "matlab_robotics_diffdrive_derivative",
    "matlab_robotics_unicycle_init", "matlab_robotics_unicycle_derivative",
    "matlab_robotics_bicycle_init", "matlab_robotics_bicycle_derivative",
    "matlab_robotics_ackermann_init", "matlab_robotics_ackermann_derivative",
    "matlab_robotics_occmap_init", "matlab_robotics_occmap_set",
    "matlab_robotics_occmap_get", "matlab_robotics_occmap_check",
    "matlab_robotics_prm_init", "matlab_robotics_prm_findpath",
    "matlab_robotics_pursuit_init", "matlab_robotics_pursuit_step",
    "matlab_robotics_collbox_init", "matlab_robotics_collsphere_init",
    "matlab_robotics_checkCollision",
    "matlab_robotics_rrt_init", "matlab_robotics_rrt_plan",
    "matlab_fusion_imu_init", "matlab_fusion_imu_step",
    "matlab_fusion_gps_init", "matlab_fusion_gps_step",
    "matlab_fusion_ahrs_init", "matlab_fusion_ahrs_step",
    "matlab_fusion_imufilter_init", "matlab_fusion_imufilter_step",
    "matlab_fusion_compfilter_init", "matlab_fusion_compfilter_step",
    "matlab_fusion_insmarg_init", "matlab_fusion_insmarg_predict",
    "matlab_fusion_insmarg_fuse_accel", "matlab_fusion_insmarg_fuse_mag",
    "matlab_fusion_insmarg_fuse_gps",
    "matlab_fusion_allanvar",
    /* ===== Navigation Toolbox — Tiers 1–4 ================================
     * Class names (occupancyMap/stateSpaceSE2/stateSpaceDubins/
     * validatorOccupancyMap/navPath/plannerRRT/plannerRRTStar/
     * plannerAStarGrid/lidarScan/lidarSLAM/poseGraph) are NOT builtins —
     * they resolve via the prelude.  Free functions + methods register here
     * (`plan`/`setOccupancy`/`getOccupancy`/`checkOccupancy`/`derivative`
     * already registered for Robotics — reused).  Method dispatch keys on
     * arg-0's pinned class in Lowering.cpp. */
    "inflate", "isStateValid", "isMotionValid",
    "sampleUniform", "matchScans", "optimizePoseGraph",
    "shortenpath", "addScan", "addRelativePose", "pathLength",
    "distance", "interpolate",
    "matlab_nav_occmap_init", "matlab_nav_occmap_set", "matlab_nav_occmap_get",
    "matlab_nav_occmap_check", "matlab_nav_occmap_inflate", "matlab_nav_occmap_setgrid",
    "matlab_nav_ss_se2_init", "matlab_nav_ss_dubins_init",
    "matlab_nav_ss_distance", "matlab_nav_ss_interpolate", "matlab_nav_ss_sample",
    "matlab_nav_validator_init", "matlab_nav_validator_isstate",
    "matlab_nav_validator_ismotion",
    "matlab_nav_path_init", "matlab_nav_path_length",
    "matlab_nav_planner_init", "matlab_nav_planner_plan", "matlab_nav_shortenpath",
    "matlab_nav_astar_init", "matlab_nav_astar_plan",
    "matlab_nav_lidarscan_init", "matlab_nav_matchscans",
    "matlab_nav_slam_init", "matlab_nav_slam_addscan",
    "matlab_nav_posegraph_init", "matlab_nav_posegraph_addrel",
    "matlab_nav_posegraph_optimize",
    /* Navigation Tiers 5-6 — localisation / reactive control / GNSS / Frenet.
     * Class names (controllerVFH/monteCarloLocalization/stateEstimatorPF/
     * gnssSensor/referencePathFrenet/trajectoryGeneratorFrenet) resolve via
     * the prelude; step/predict/correct reused from fusion/ident. */
    "initialize", "getStateEstimate",
    "global2frenet", "frenet2global", "connect",
    "gnssconstellation", "pseudoranges", "receiverposition",
    "matlab_nav_vfh_step",
    "matlab_nav_mcl_init", "matlab_nav_mcl_step",
    "matlab_nav_pf_initialize", "matlab_nav_pf_predict",
    "matlab_nav_pf_correct", "matlab_nav_pf_estimate",
    "matlab_nav_gnss_step", "matlab_nav_gnssconstellation",
    "matlab_nav_pseudoranges", "matlab_nav_receiverposition",
    "matlab_nav_frenet_init", "matlab_nav_frenet_g2f", "matlab_nav_frenet_f2g",
    "matlab_nav_trajgen_init", "matlab_nav_trajgen_connect",
    /* ===== Deep Learning Toolbox — Tiers 1-2 (dlarray + autodiff) =========
     * `dlarray` is a class (resolves via prelude, NOT registered).  The
     * activation/loss free-function names + extractdata + dlgradient register
     * here; tanh/sum/mean/log/exp are already builtins (the Lowering arm only
     * reroutes them when the argument is a dlarray). */
    "relu", "sigmoid", "softmax", "crossentropy", "mse", "lstm", "embed",
    "gru", "bilstm", "lstmp",
    /* DL HDL Tier H1 — INT8 quantization (plain numeric in/out). */
    "dlquantize", "dlqscale",
    "extractdata", "dlgradient",
    "matlab_dlnet_dlarray_init", "matlab_dlnet_extractdata",
    "matlab_dlnet_plus", "matlab_dlnet_minus", "matlab_dlnet_mtimes",
    "matlab_dlnet_times",
    "matlab_dlnet_relu", "matlab_dlnet_sigmoid", "matlab_dlnet_tanh",
    "matlab_dlnet_softmax", "matlab_dlnet_sum", "matlab_dlnet_mean",
    "matlab_dlnet_log", "matlab_dlnet_exp",
    "matlab_dlnet_crossentropy", "matlab_dlnet_mse", "matlab_dlnet_lstm",
    "matlab_dlnet_transpose", "matlab_dlnet_embed",
    "matlab_dlnet_gru", "matlab_dlnet_bilstm", "matlab_dlnet_lstmp",
    "matlab_dlnet_quantize", "matlab_dlnet_qscale",
    "matlab_dlnet_grad", "matlab_dlnet_reset",
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
    /* Optimization Toolbox — see docs/optim_toolbox_roadmap.md.
     * Tier-1.1: scalar root finder via Brent's method.  Two operand
     * shapes share the name `fzero` and are split by the lowering
     * pass — `fzero(@fn, x0)` (scalar guess) vs `fzero(@fn, [a b])`
     * (bracket).  Tier-1.2–1.9: 1-D minimiser (`fminbnd`), N-D
     * minimisers (`fminsearch` Nelder–Mead, `fminunc` BFGS), linear
     * programming (`linprog`, 3- or 7-arg), non-negative least
     * squares (`lsqnonneg`), and the scalar-equation solver
     * (`fsolve`).  All single-return in Tier-1. */
    "fzero", "fminbnd", "fminsearch", "fminunc",
    "linprog", "lsqnonneg", "fsolve",
    /* Tier-2: constrained nonlinear (`fmincon`), convex QP
     * (`quadprog`), constrained linear least squares (`lsqlin`),
     * nonlinear least squares (`lsqnonlin`), and curve fitting
     * (`lsqcurvefit`).  `fsolve` (registered above) also gains an
     * N-D vector form in Tier-2.  All single-return. */
    "fmincon", "quadprog", "lsqlin", "lsqnonlin", "lsqcurvefit",
    /* Tier-3: mixed-integer LP (`intlinprog`), second-order cone
     * programming (`coneprog`), minimax (`fminimax`), goal attainment
     * (`fgoalattain`), and semi-infinite programming (`fseminf`). */
    "intlinprog", "coneprog", "fminimax", "fgoalattain", "fseminf",
    /* Tier-4: problem-based API expression-DAG builders.  The
     * MATLAB-facing `optimvar` / `optimproblem` are user functions in
     * runtime/optim_classdefs.m; the `matlab_optim_pb_*` family below
     * is the runtime the classdef methods forward to. */
    "matlab_optim_pb_var", "matlab_optim_pb_const",
    "matlab_optim_pb_add", "matlab_optim_pb_sub", "matlab_optim_pb_neg",
    "matlab_optim_pb_mul", "matlab_optim_pb_div", "matlab_optim_pb_pow",
    "matlab_optim_pb_le", "matlab_optim_pb_ge", "matlab_optim_pb_eq",
    "matlab_optim_pb_solve",
    /* Tier-5: problem-based equation solving (`eqnproblem`). */
    "matlab_optim_pb_solve_eqn",
    /* Partial Differential Equation Toolbox — see docs/pde_toolbox_roadmap.md.
     * Function-form numeric core; the MATLAB-side createpde/femodel
     * wrappers (Tier-3 polish) layer kwarg sugar on top. */
    "pde_mesh_rect_tri", "pde_boundary_nodes_rect",
    "pde_assemble_poisson_2d", "pde_apply_dirichlet",
    "pde_mesh_cuboid_tet", "pde_face_nodes",
    "pde_assemble_elast_3d", "pde_face_pressure_3d",
    "pde_apply_fixed_3d", "pde_reshape_disp_3d",
    "pde_von_mises_3d", "pde_node_von_mises_3d", "pde_peak_disp_3d",
    "pde_sys_K", "pde_sys_F", "pde_sys_M",
    "pde_mesh_nodes", "pde_mesh_triangles",
    "pde_mesh_tets", "pde_mesh_faces",
    /* Tier-3 transient + modal. */
    "pde_assemble_transient_2d", "pde_eigsmall",
    "pde_step_forward_euler_2d", "pde_init_uniform_2d",
    /* Tier-4 nonlinear. */
    "pde_solve_nonlinear_2d", "pde_result_solution",
    "pde_result_num_iters", "pde_result_resid",
    /* PDE geometry importers (STL ASCII + binary, GLB / glTF 2.0). */
    "pde_load_stl", "pde_load_glb", "pde_save_stl",
    /* Sparse matrix infrastructure — see runtime/runtime_sparse.cpp. */
    "sparse", "spnnz", "sprows", "spcols", "speye", "spdiag",
    "sparse_matvec", "spfull",
    "pcg", "pcg_x", "pcg_flag", "pcg_relres", "pcg_iter",
    /* Sparse FEM assembly. */
    "pde_assemble_poisson_2d_sparse",
    "pde_apply_dirichlet_sparse",
    "pde_assemble_elast_3d_sparse",
    "pde_apply_fixed_3d_sparse",
    "pde_sys_K_sparse",
    /* Surface→tet voxelizer. */
    "pde_voxelize_surface",
    /* femodel classdef façade kernel + setters. */
    "pde_solve_femodel", "pde_solve",
    "pde_kernel_mesh", "pde_kernel_u", "pde_kernel_vm",
    "pde_set_material", "pde_set_face_fixed", "pde_set_face_pressure",
    "pde_generate_mesh",
    /* Geometry primitives — voxelize-AABB family. */
    "pde_multicylinder", "pde_multicylinder_hollow", "pde_multisphere",
    "pde_translate", "pde_rotate", "pde_scale",
    /* Tier-3 scalar AnalysisType: thermal + electrostatic. */
    "pde_set_face_temperature", "pde_set_face_heat",
    "pde_set_face_voltage", "pde_set_face_charge",
    "pde_set_body_heat", "pde_set_body_charge",
    "pde_solve_thermal_steady", "pde_solve_electrostatic",
    "pde_solve_magnetostatic", "pde_solve_dc_conduction",
    "pde_set_face_potential", "pde_set_face_current", "pde_set_body_current",
    "pde_solve_structural_transient",
    "pde_set_time_step", "pde_set_num_steps",
    "pde_kernel_uhist", "pde_kernel_tlist",
    "pde_solve_structural_modal",
    "pde_set_num_modes",
    "pde_kernel_freqs",
    "pde_eig_lanczos_si",
    "pde_eig_lanczos_si_full", "pde_eig_lambda", "pde_eig_phi",
    "pde_solve_structural_frequency", "pde_set_freq_list",
    "pde_kernel_freqlist",
    "pde_solve_harmonic_em", "pde_set_wave_number",
    /* Modal superposition transient. */
    "pde_solve_structural_transient_modal",
    "pde_set_rayleigh", "pde_set_modal_results",
    /* Sparse Krylov: MINRES (unpreconditioned indefinite),
     * GMRES + ILU(0) (general indefinite, production solver). */
    "minres",
    "sparse_gmres_ilu0",
    /* T10 quadratic-tet path. */
    "pde_mesh_quadratic", "pde_assemble_elast_3d_t10",
    "pde_face_pressure_3d_t10", "pde_face_nodes_t10",
    "pde_node_von_mises_3d_t10",
    "pde_apply_fixed_3d_t10",
    /* thermalTransient + thermal-stress coupling + Picard nonconstant-k. */
    "pde_solve_thermal_transient",
    "pde_set_initial_temperature",
    "pde_set_cell_temperature",
    "pde_set_reference_temperature",
    /* Tier-4: ROM + adaptmesh + structuralStaticNL + multi-component. */
    "pde_reduce", "reduce",
    "pde_reconstruct_solution", "reconstructSolution",
    "pde_refine_mesh", "refineMesh",
    "pde_adapt_mesh",  "adaptmesh",
    "pde_solve_structural_static_nl",
    "pde_set_multi_coeff", "pde_solve_multi",
    "pde_multi_u", "pde_multi_v",
    /* Tier-4 closure: Craig-Bampton + Total-Lagrangian Newton +
     * Bey refinement + N-component PDEs. */
    "pde_set_interface_face",
    "pde_reduce_craig_bampton",
    "pde_solve_structural_static_tl",
    "pde_refine_mesh_bey", "refineMeshBey",
    "pde_adapt_mesh_marked",
    "pde_set_multi_coeff_n", "pde_solve_multi_n", "pde_multi_n_u",
    /* MATLAB-faithful legacy aliases. */
    "solvepde", "solvepdeeig",
    "pdegplot", "pdemesh",
    "specifyCoefficients", "applyBoundaryCondition",
    "pde_assemble_poisson_3d_sparse", "pde_apply_dirichlet_3d_sparse",
    "pde_face_scalar_load_3d",
    /* pdeplot3D rendering — Cairo unstructured-mesh painter. */
    "pdeplot3d", "pdeplot3D",
    "pdeplot3d_deformation", "pdeplot3d_deform_scale",
    "pdeplot",
    "nargin", "nargout", "varargin", "varargout",
    "fopen", "fclose", "fgetl", "feof",
    "fread", "fwrite",
    "readtable", "readmatrix",
    "save", "load",
    // §17.5 #6 — cross-dialect composition: call a signal-flow
    // .mflow from a .m program. Lowers to `matlab_mflowlink_run`
    // (see runtime/runtime_mflowlink_call.cpp).
    "mflowlink_run",
    "conj", "real", "imag", "angle", "complex",
    "fft", "ifft", "fft2", "ifft2",
    "conv", "conv2",
    "filter", "any", "all", "tril", "triu",
    "fftshift", "ifftshift",
    "std", "var", "median", "diff",
    "meshgrid", "ndgrid", "peaks",
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
    "figure", "gcf", "gca", "close",
    "plot", "plot3", "bar", "scatter", "stem", "stairs", "area",
    "errorbar", "histogram",
    "imshow", "imagesc", "pcolor", "surf", "surfc", "mesh", "meshc", "contour",
    "title", "xlabel", "ylabel", "zlabel", "legend", "text",
    "colorbar", "colormap",
    "grid", "hold", "axis", "box", "xlim", "ylim", "view",
    "loglog", "semilogx", "semilogy",
    "subplot", "saveas", "print",
    "xline", "yline",
    "xticks", "yticks", "xticklabels", "yticklabels",
    "yyaxis", "contourf", "quiver",
    /* IDE / REPL actions accepted as no-ops outside the REPL so
     * scripts that include them still compile. Roadmap-tracked in
     * docs/plotting.md / docs/roadmap.md. */
    "clc",
    /* Tier-3 plotting roadmap items — accepted as no-ops in the
     * headless renderer.  `surfl` / `shading` / `material` /
     * `camlight` / `lighting` need a 3-D rasterizer with light
     * vectors + per-vertex normals (Tier-5 "won't make sense
     * headless" per docs/plotting.md §3).  Registered here so
     * scripts don't error; they emit nothing in the PNG output. */
    "surfl", "shading", "material", "camlight", "lighting",
    /* `set` is the universal MATLAB property setter
     * `set(handle, 'Name', value, ...)`.  Accepted as a no-op until
     * the Name-Value plumbing arc lands (docs/roadmap.md
     * "Name-value-on-unknown-handle bail"). */
    "set",
    /* Function-form colormap getters: `magma(N)` returns an N×3
     * RGB matrix.  Sema-registered so calls compile; the lowering
     * synthesises the table via the shipped colormap surface. */
    "magma", "inferno", "plasma", "cividis", "turbo",
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
    "fskmod", "fskdemod",
    /* ANT-Tier-2 — wire-antenna MoM (Antenna MVP). */
    "antennaWireSolve", "antennaWirePattern", "antennaWireSparameters",
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
    /* Tier-4: analog sources + comparators + Schmitt triggers. */
    "writeVerilogASource", "writeVerilogAComparator", "writeVerilogASchmitt",
    /* Tier-5: VCO via idtmod. */
    "writeVerilogAVCO",
    /* Tier-6: behavioral DAC. */
    "writeVerilogADAC",
    /* Tier-7: compact components + sensor models. */
    "writeVerilogADiode", "writeVerilogAOpAmp",
    "writeVerilogARTD",   "writeVerilogAThermistor",
    /* Tier-8: white + flicker noise sources. */
    "writeVerilogANoise",
    /* Tier-9: lookup tables via $table_model. */
    "writeVerilogATable",
    /* Tier-7 follow-on: composite RF / signal-chain blocks. */
    "writeVerilogAAmplifier", "writeVerilogAAM", "writeVerilogAIQMod",
    /* GPU Coder pragma stubs.  Folded from `coder.gpu.kernelfun()` /
     * `coder.gpu.kernel` / `coder.gpu.constantMemory(X)` /
     * `coder.gpu.stream()` / `coder.gpu.sync()` by parsePostfix.
     * Recognised + intercepted in Lowering::lowerStmt (ExprStmt arm) so
     * they emit no IR; declared here so Sema accepts the name. */
    "coder_gpu_kernelfun", "coder_gpu_kernel",
    "coder_gpu_constantMemory", "coder_gpu_stream", "coder_gpu_sync",
    /* GPU Coder helper functions.  Folded from `gpucoder.reduce(X,@f)` /
     * `gpucoder.matrixMatrixKernel(C,op,A,B)` / `gpucoder.sort` /
     * `gpucoder.batchedMatMul` / `gpucoder.stencilKernel` (deprecated). */
    "gpucoder_reduce", "gpucoder_matrixMatrixKernel",
    "gpucoder_sort", "gpucoder_batchedMatMul", "gpucoder_stencilKernel",
    /* Phase 4 of lapack_roadmap §4 — GPU library replacement for GEMM.
     * `gpucoder.gemm(A,B)` folds to `gpucoder_gemm`; the runtime
     * dispatcher (matlab_gpu_gemm) routes to MPSMatrixMultiplication
     * on Metal / cuBLAS sgemm on CUDA (future) / falls back to the CPU
     * BLAS path otherwise. */
    "gpucoder_gemm",
    /* `coder.gpuConfig('mex' | 'lib' | 'exe' | 'dll')` returns a
     * config record consumed by the -emit-{cuda,metal,opencl} lanes. */
    "coder_gpuConfig",
    /* GPU Coder host-side carriers — `gpuArray` is a classdef pulled
     * in by the prelude (gpu_classdefs.m); the symbols listed below
     * are the free functions declared alongside it (gather, etc.).
     * Registering as builtin so Sema accepts the name in source code
     * before the classdef prelude wires it up.  Note: gpuArray itself
     * is intentionally NOT in this list — it's a class binding. */
    "gather", "existsOnGPU", "gpuDevice",
    /* Runtime ABI shims called from gpu_classdefs.m method bodies. */
    "matlab_gpu_upload", "matlab_gpu_download",
    "matlab_gpu_exists_on_gpu", "matlab_gpu_active_target_name",
    "matlab_gpu_device_name",
    /* gpuArray.<static> factory names (folded by Parser).  CPU-debug
     * lane routes each to the corresponding matlab_* allocator. */
    "gpuArray_rand", "gpuArray_randn", "gpuArray_randi",
    "gpuArray_zeros", "gpuArray_ones", "gpuArray_eye",
    "gpuArray_true", "gpuArray_false",
    "gpuArray_linspace", "gpuArray_colon",
    "gpuArray_inf", "gpuArray_nan",
    /* PCT names that may appear in the validation suite. */
    "gpuDeviceCount", "wait", "tic", "toc",
    /* Stencil v2 — `stencilfun(@f, A, [m n])` is the supported form
     * (matched as a regular builtin call; not parser-folded). */
    "stencilfun",
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
  // Re-runnable Resolver invariant. The monomorphizer (#38 Phases 3+4)
  // re-invokes resolve() on the same TU after appending clones; without
  // these clears, F.ParamRefs / F.OutputRefs would accumulate duplicate
  // bindings from each prior run and downstream consumers would see
  // stale entries first. Single-pass callers see no behavior change
  // since these vectors start empty on a freshly-parsed Function.
  F.ParamRefs.clear();
  F.OutputRefs.clear();
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

void Resolver::applyWorkspaceKind(Binding *NB, std::string_view Name,
                                  Scope *S) {
  /* REPL cross-input persistence: when the resolver auto-declares a
   * name that wasn't assigned in this TU, query the live workspace for
   * the kind under which a prior input bound it.  Stamping
   * InferredType lets every downstream dispatch site (string disp,
   * strlen, isstring, the workspace load path itself) see the correct
   * shape — without this, a fresh `disp(t)` after an earlier
   * `t = "..."` couldn't tell matrix from string and either silently
   * dropped the read or matrix-cast the descriptor's bytes. */
  if (!ReplMode || !WorkspaceKindHook) return;
  int K = WorkspaceKindHook(Name.data(), (int64_t)Name.size());
  /* Kind encoding (matches matlab_dbg_ws_kind):
   *   0 = f64 scalar, 1 = matlab_mat*, 2 = matlab_obj*,
   *   3 = matlab_string*, 4 = matlab_mat_u8*, 5 = matlab_mat_i32*.
   *   Stamp InferredType for the kinds the lowering can specialise on
   *   — without a scalar-double stamp, the workspace load path falls
   *   back to matlab_ws_get_mat (returns ptr) and downstream consumers
   *   that need an f64 (num2str / arithmetic / range bounds) silently
   *   misbehave.  Kinds 4/5 (typed-int matrices) are stamped with a
   *   non-scalar Array(UInt8/Int32, unknown) so the BinaryOp emit site
   *   picks the typed runtime entry points. */
  if (K == 0) NB->InferredType = TC.scalar(Dtype::Double);
  /* Kind 1 = generic matlab_mat* (real double matrix). Without this
   * stamp, downstream MLIR lowerings fall back to a ptr-typed
   * workspace load and elementwise operations (`.*`, matrix `+ -`,
   * `norm`, etc.) misdispatch on the next REPL turn. Stamp the
   * canonical "double array of unknown shape" so the BinaryOp /
   * call_builtin path picks the matrix-mat runtime entries. */
  else if (K == 1) NB->InferredType = TC.arrayOf(Dtype::Double, Shape::unknown());
  else if (K == 3) NB->InferredType = TC.stringScalar();
  else if (K == 4) NB->InferredType = TC.arrayOf(Dtype::UInt8, Shape::unknown());
  else if (K == 5) NB->InferredType = TC.arrayOf(Dtype::Int32, Shape::unknown());
  /* Kind 7 = matlab_sym* (Phase 6 — Symbolic Math Toolbox).  No
   * InferredType is set (Sema has no SymType yet), but the binding is
   * tagged so the MLIR lowering's BinaryOp / disp / workspace-store
   * dispatch sees it as sym across TUs. */
  else if (K == 7) NB->IsSym = true;
  /* Kind 8 = matlab_symmat* (Phase 6.1 — symbolic matrix). */
  else if (K == 8) NB->IsSymmat = true;
  /* Kind 12 = matlab_struct* (plain field-holder, not a classdef
   * instance).  Stamping IsStruct lets the MLIR lowering re-populate
   * StructInitialised on the next compile so `lb.PathLoss` and friends
   * route through the struct-get path. */
  else if (K == 12) NB->IsStruct = true;
  /* Kind 2 = matlab_obj* (classdef instance).  Re-pin the binding to
   * the runtime-tracked class so the obj-call sugar / dot-method
   * dispatch / class operator overloads stay live across REPL turns.
   * Without this, a cross-input `crc(1)` (when `crc` was a
   * SystemObject set in a prior turn) would fall through to the
   * matrix-subscript path and read garbage.  The classdef itself must
   * be in scope by this point — the REPL prelude loader pulls it in
   * via the same workspace-scan path. */
  else if (K == 2 && WorkspaceClassNameHook) {
    int64_t CnLen = 0;
    const char *Cn = WorkspaceClassNameHook(Name.data(),
                                            (int64_t)Name.size(), &CnLen);
    if (Cn && CnLen > 0) {
      std::string_view CnView(Cn, (size_t)CnLen);
      Binding *CB = S ? S->lookup(CnView) : nullptr;
      if (CB && CB->Kind == BindingKind::Class && CB->ClassDef)
        NB->PinnedClass = CB->ClassDef;
    }
  }
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
          Binding *NB = Sema.newBinding();
          Binding *B = FnScope->getOrDeclareVar(N->Name, NB);
          /* Compound LHS (`prob.X = ...`, `prob(i) = ...`): the root
           * name is read-modified, not freshly rebound. In REPL mode a
           * freshly auto-declared root must be rehydrated from the
           * workspace so e.g. `prob` re-pins to its classdef and the
           * field store routes through matlab_obj_set_* instead of
           * spawning a throwaway struct. Bare `x = ...` (Root == L) is
           * a clean rebind — pinnedOfRhs handles its class pin. */
          if (B == NB && Root != L)
            applyWorkspaceKind(NB, N->Name, FnScope);
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
    // A null LHS slot is the `~` ignore-output placeholder (`[~, idx] = …`).
    for (Expr *L : A.LHS) if (L) resolveLValue(*L, S);
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
        [&pinnedOfRhs, S](Expr *RE) -> ClassDef * {
      if (!RE) return nullptr;
      if (auto *NE = dynamic_cast<NameExpr *>(RE)) {
        if (NE->Ref && NE->Ref->PinnedClass) return NE->Ref->PinnedClass;
        return nullptr;
      }
      if (auto *CX = dynamic_cast<CallOrIndex *>(RE)) {
        if (auto *NX = dynamic_cast<NameExpr *>(CX->Callee)) {
          if (NX->Ref && NX->Ref->Kind == BindingKind::Class &&
              NX->Ref->ClassDef) return NX->Ref->ClassDef;
          /* Optimization Toolbox problem-based factory functions —
           * `optimvar` / `optimintvar` produce an OptimizationExpression
           * and `optimproblem` an OptimizationProblem.  These are
           * resolved by name (the classdef prelude is *appended* after
           * the user script, so the generic "user function returning a
           * pinned output" rule below can't see the pin yet when the
           * caller is resolved). */
          auto classByName = [S](const char *Nm) -> ClassDef * {
            if (Binding *B = S ? S->lookup(Nm) : nullptr)
              if (B->Kind == BindingKind::Class && B->ClassDef)
                return B->ClassDef;
            return nullptr;
          };
          if (NX->Name == "optimvar" || NX->Name == "optimintvar") {
            if (ClassDef *C = classByName("OptimizationExpression"))
              return C;
          }
          if (NX->Name == "optimproblem") {
            if (ClassDef *C = classByName("OptimizationProblem"))
              return C;
          }
          if (NX->Name == "eqnproblem") {
            if (ClassDef *C = classByName("EquationProblem"))
              return C;
          }
          /* MPC Tier-1: the `mpc(plant, p, m)` and `mpcstate(nx, nu)`
           * factories return class-pinned instances.  Without these
           * hooks, downstream `mpcmove(obj, ...)` / `sim(obj, ...)`
           * dispatches in Lowering.cpp can't see the class pin and
           * the call-builtin emit site falls through to the classdef
           * function body — which is brittle because the formal
           * parameter slots carry `none` types. */
          if (NX->Name == "mpc") {
            if (ClassDef *C = classByName("mpc")) return C;
          }
          if (NX->Name == "mpcstate") {
            if (ClassDef *C = classByName("mpcstate")) return C;
          }
          if (NX->Name == "mpcmoveopt") {
            if (ClassDef *C = classByName("mpcmoveopt")) return C;
          }
          if (NX->Name == "mpcsimopt") {
            if (ClassDef *C = classByName("mpcsimopt")) return C;
          }
          if (NX->Name == "explicitMPC" || NX->Name == "generateExplicitMPC") {
            if (ClassDef *C = classByName("explicitMPC")) return C;
          }
          if (NX->Name == "nlmpc") {
            if (ClassDef *C = classByName("nlmpc")) return C;
          }
          /* System Identification Tier-1: the `arx` / `ar` estimators
           * return a class-pinned `idpoly`.  Without this, downstream
           * `compare(z, m)` / `sim(m, u)` / `m.A` dispatches can't see
           * the class pin (the classdef prelude is appended after the
           * user script, so the generic "user function returning a
           * pinned output" rule below can't resolve it yet). */
          if (NX->Name == "arx" || NX->Name == "ar" ||
              NX->Name == "armax" || NX->Name == "oe" || NX->Name == "bj" ||
              NX->Name == "iv4" || NX->Name == "tfest" ||
              NX->Name == "impulseest") {
            if (ClassDef *C = classByName("idpoly")) return C;
          }
          /* Tier-3: n4sid / ssest return a class-pinned idss. */
          if (NX->Name == "n4sid" || NX->Name == "ssest") {
            if (ClassDef *C = classByName("idss")) return C;
          }
          /* Tier-4: greyest returns a class-pinned idgrey. */
          if (NX->Name == "greyest") {
            if (ClassDef *C = classByName("idgrey")) return C;
          }
          /* Tier-4: etfe / spa return a class-pinned idfrd. */
          if (NX->Name == "etfe" || NX->Name == "spa") {
            if (ClassDef *C = classByName("idfrd")) return C;
          }
          /* Tier-5: nlgreyest returns a class-pinned idnlgrey. */
          if (NX->Name == "nlgreyest") {
            if (ClassDef *C = classByName("idnlgrey")) return C;
          }
          /* Tier-6: arxOptions() returns a class-pinned arxOptions. */
          if (NX->Name == "arxOptions") {
            if (ClassDef *C = classByName("arxOptions")) return C;
          }
          /* Global Optim Tier-2: MultiStart() / GlobalSearch() factories
           * return class-pinned solver objects (so run(solver,…)
           * dispatches on the pinned class). */
          if (NX->Name == "MultiStart") {
            if (ClassDef *C = classByName("MultiStart")) return C;
          }
          if (NX->Name == "GlobalSearch") {
            if (ClassDef *C = classByName("GlobalSearch")) return C;
          }
          /* Stats Tier-1: makedist / fitdist return a class-pinned
           * ProbDistUnivParam distribution object. */
          if (NX->Name == "makedist" || NX->Name == "fitdist") {
            if (ClassDef *C = classByName("ProbDistUnivParam")) return C;
          }
          /* Stats Tier-3: fitlm / fitglm return a class-pinned LinearModel. */
          if (NX->Name == "fitlm" || NX->Name == "fitglm") {
            if (ClassDef *C = classByName("LinearModel")) return C;
          }
          /* Image Tier-3: fitgeotform2d returns a class-pinned affine2d. */
          if (NX->Name == "fitgeotform2d") {
            if (ClassDef *C = classByName("affine2d")) return C;
          }
          /* Stats Tier-5: fitc* return a class-pinned ClassificationModel. */
          if (NX->Name == "fitcknn" || NX->Name == "fitcnb" ||
              NX->Name == "fitcdiscr" || NX->Name == "fitctree" ||
              NX->Name == "fitcsvm" || NX->Name == "fitcecoc" ||
              NX->Name == "fitcensemble" || NX->Name == "TreeBagger") {
            if (ClassDef *C = classByName("ClassificationModel")) return C;
          }
          /* Curve Fitting Tier-1: fit(x,y,'polyN') returns a class-pinned
           * cfit.  Without this, the LHS slot of `f = fit(...)` (and the
           * first slot of `[f,gof] = fit(...)`) isn't pinned and the
           * `f(xq)` / `feval(f,xq)` / `coeffvalues(f)` dispatches fail. */
          if (NX->Name == "fit") {
            /* A 2-digit poly tag ('poly23') in the model arg means a surface
             * fit → sfit; anything else is a curve → cfit. */
            bool isSurface = false;
            if (CX->Args.size() >= 3) {
              std::string mt;
              if (auto *CL = dynamic_cast<CharLiteral *>(CX->Args[2])) mt = CL->Value;
              else if (auto *SL = dynamic_cast<StringLiteral *>(CX->Args[2])) mt = SL->Value;
              if (mt.size() == 6 && mt.compare(0, 4, "poly") == 0 &&
                  isdigit(static_cast<unsigned char>(mt[4])) &&
                  isdigit(static_cast<unsigned char>(mt[5])))
                isSurface = true;
            }
            if (ClassDef *C = classByName(isSurface ? "sfit" : "cfit")) return C;
          }
          if (NX->Name == "fitoptions") {
            if (ClassDef *C = classByName("fitoptions")) return C;
          }
          if (NX->Name == "fittype") {
            if (ClassDef *C = classByName("fittype")) return C;
          }
          /* Curve Fitting Tier-6: spline / pchip / ppmak / fnder / fnint
           * each yield a ppform; the pin lets fnval(pp,xx) and fnder(pp)
           * dispatch correctly. */
          if (NX->Name == "spline" || NX->Name == "pchip" ||
              NX->Name == "ppmak" || NX->Name == "fnder" || NX->Name == "fnint") {
            if (ClassDef *C = classByName("ppform")) return C;
          }
          /* User function whose body returns a class-pinned value —
           * propagate that pin to the caller's LHS. */
          if (NX->Ref && NX->Ref->Kind == BindingKind::Function &&
              NX->Ref->FuncDef) {
            Function *F = NX->Ref->FuncDef;
            if (!F->OutputRefs.empty() && F->OutputRefs.front() &&
                F->OutputRefs.front()->PinnedClass)
              return F->OutputRefs.front()->PinnedClass;
          }
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
            /* Econometrics: `EstMdl = estimate(Mdl, y)` returns the same
             * model class as the template, so downstream forecast/infer/
             * simulate dispatch (which keys on the receiver class) fires. */
            if (Arg0Cls && NX->Name == "estimate" &&
                (Arg0Cls->Name == "arima" || Arg0Cls->Name == "garch" ||
                 Arg0Cls->Name == "egarch" || Arg0Cls->Name == "gjr" ||
                 Arg0Cls->Name == "varm" || Arg0Cls->Name == "ssm" ||
                 Arg0Cls->Name == "dssm" || Arg0Cls->Name == "bayeslm"))
              return Arg0Cls;
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
      /* Only the FIRST LHS receives the class pin.  A class factory
       * yields its instance as output 0; any trailing outputs of a
       * multi-return form (e.g. `[f, gof] = fit(...)` where gof is a
       * plain struct) must not inherit the pin or their field access
       * would mis-route through class-property dispatch. */
      if (!A.LHS.empty()) {
        if (auto *LN = dynamic_cast<NameExpr *>(A.LHS.front())) {
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
        applyWorkspaceKind(NB, N.Name, S);
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
