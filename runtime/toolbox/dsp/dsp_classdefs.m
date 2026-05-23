% dsp_classdefs.m — DSP System Toolbox System-Object surface.
%
% These classdefs are auto-prepended by matlabc when the user input
% mentions a `dsp.*` constructor (the parser folds `dsp.FIRFilter` to the
% flat class name `dsp_FIRFilter`; see lib/Parse/Parser.cpp).
%
% Each object is a `handle` so its internal state (the tapped-delay line,
% per-section biquad words, polyphase commutator, adaptive weights)
% persists by reference across the frame-based `y = obj(frame)` calls —
% which the compiler lowers to `step(obj, frame)` (Lowering.cpp).
%
% The `step` / `reset` method bodies are deliberately one-liners that
% forward the receiver `obj` to the `matlab_dsp_*` runtime entries in
% runtime/toolbox/dsp/runtime_dsp.cpp: a value read from `obj.Prop` inside
% a method body is untyped, so the compute + the in-place state mutation
% run in C++ (the System-Identification EKF/RLS precedent).  Constructors
% likewise avoid matrix ops (numel/size) on the arguments — they only
% *store* the coefficient property, then call `matlab_dsp_init_state`,
% which sizes the State buffer from that property in the runtime.

% ===================================================================
% Tier-1 — core filter objects
% ===================================================================

classdef dsp_FIRFilter < handle
    properties
        Numerator
        State
        IsLocked
    end
    methods
        function obj = dsp_FIRFilter(a, b)
            obj.IsLocked = 0;
            obj.Numerator = 1;
            obj.State = zeros(1, 1);
            if nargin == 2
                obj.Numerator = b;       % name-value: ('Numerator', b)
            elseif nargin == 1
                obj.Numerator = a;       % positional: (b)
            end
            matlab_dsp_init_state(obj);
        end
        function y = step(obj, x)
            y = matlab_dsp_iir_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function s = getDiscreteState(obj)
            s = matlab_dsp_get_state(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end

classdef dsp_IIRFilter < handle
    properties
        Numerator
        Denominator
        State
        IsLocked
    end
    methods
        function obj = dsp_IIRFilter(a, b, c, d)
            obj.IsLocked = 0;
            obj.Numerator = 1;
            obj.Denominator = 1;
            obj.State = zeros(1, 1);
            if nargin == 4
                obj.Numerator = b;       % ('Numerator', b, 'Denominator', d)
                obj.Denominator = d;
            elseif nargin == 2
                obj.Numerator = a;       % positional: (b, a)
                obj.Denominator = b;
            elseif nargin == 1
                obj.Numerator = a;
            end
            matlab_dsp_init_state(obj);
        end
        function y = step(obj, x)
            y = matlab_dsp_iir_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function s = getDiscreteState(obj)
            s = matlab_dsp_get_state(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end

classdef dsp_SOSFilter < handle
    properties
        SOSMatrix
        ScaleValues
        State
        IsLocked
    end
    methods
        function obj = dsp_SOSFilter(a, b)
            obj.IsLocked = 0;
            obj.ScaleValues = 1;
            obj.SOSMatrix = [1 0 0 1 0 0];
            obj.State = zeros(1, 2);
            if nargin == 2
                obj.SOSMatrix = b;       % ('SOSMatrix', sos)
            elseif nargin == 1
                obj.SOSMatrix = a;       % positional: (sos)
            end
            matlab_dsp_init_state(obj);
        end
        function y = step(obj, x)
            y = matlab_dsp_sos_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function s = getDiscreteState(obj)
            s = matlab_dsp_get_state(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end

classdef dsp_BiquadFilter < handle
    properties
        SOSMatrix
        ScaleValues
        State
        IsLocked
    end
    methods
        function obj = dsp_BiquadFilter(a, b)
            obj.IsLocked = 0;
            obj.ScaleValues = 1;
            obj.SOSMatrix = [1 0 0 1 0 0];
            obj.State = zeros(1, 2);
            if nargin == 2
                obj.SOSMatrix = b;
            elseif nargin == 1
                obj.SOSMatrix = a;
            end
            matlab_dsp_init_state(obj);
        end
        function y = step(obj, x)
            y = matlab_dsp_sos_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end

% ===================================================================
% Tier-3 — adaptive filter objects
% ===================================================================

classdef dsp_LMSFilter < handle
    properties
        Length
        StepSize
        Method        % 0 = LMS, 1 = NLMS
        Weights
        TapState
        IsLocked
    end
    methods
        function obj = dsp_LMSFilter(a, b, c, d)
            obj.IsLocked = 0;
            obj.Length = 32;
            obj.StepSize = 0.1;
            obj.Method = 0;
            % Weights / TapState are sized lazily by the runtime step from
            % Length, so the constructor never touches matrices.
            obj.Weights = 0;
            obj.TapState = 0;
            if nargin == 1
                obj.Length = a;                  % positional (L)
            elseif nargin == 2
                obj.Length = b;                  % ('Length', L)
            elseif nargin == 4
                obj.Length = b;                  % ('Length', L, 'StepSize', mu)
                obj.StepSize = d;
            end
        end
        function e = step(obj, x, d)
            e = matlab_dsp_lms_step(obj, x, d);
        end
        function w = getWeights(obj)
            w = matlab_dsp_get_weights(obj);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end

classdef dsp_RLSFilter < handle
    properties
        Length
        ForgettingFactor
        Weights
        TapState
        InvCov
        IsLocked
    end
    methods
        function obj = dsp_RLSFilter(a, b, c, d)
            obj.IsLocked = 0;
            obj.Length = 32;
            obj.ForgettingFactor = 1;
            obj.Weights = 0;
            obj.TapState = 0;
            obj.InvCov = 0;
            if nargin == 1
                obj.Length = a;
            elseif nargin == 2
                obj.Length = b;                  % ('Length', L)
            elseif nargin == 4
                obj.Length = b;
                obj.ForgettingFactor = d;        % ('Length', L, 'ForgettingFactor', lam)
            end
        end
        function e = step(obj, x, d)
            e = matlab_dsp_rls_step(obj, x, d);
        end
        function w = getWeights(obj)
            w = matlab_dsp_get_weights(obj);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end

% ===================================================================
% Tier-4 — multirate + multistage System Objects
% ===================================================================

classdef dsp_FIRDecimator < handle
    properties
        DecimationFactor
        Numerator
        State
        IsLocked
    end
    methods
        function obj = dsp_FIRDecimator(a, b, c, d)
            obj.IsLocked = 0;
            obj.DecimationFactor = 2;
            obj.Numerator = 1;
            obj.State = zeros(1, 1);
            if nargin == 2
                obj.DecimationFactor = a;       % (M, b) positional
                obj.Numerator = b;
            elseif nargin == 1
                obj.DecimationFactor = a;
            elseif nargin == 4
                obj.DecimationFactor = b;       % ('DecimationFactor',M,'Numerator',b)
                obj.Numerator = d;
            end
            matlab_dsp_init_state(obj);
        end
        function y = step(obj, x)
            y = matlab_dsp_firdecim_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end

classdef dsp_FIRInterpolator < handle
    properties
        InterpolationFactor
        Numerator
        State
        IsLocked
    end
    methods
        function obj = dsp_FIRInterpolator(a, b, c, d)
            obj.IsLocked = 0;
            obj.InterpolationFactor = 2;
            obj.Numerator = 1;
            obj.State = zeros(1, 1);
            if nargin == 2
                obj.InterpolationFactor = a;    % (L, b) positional
                obj.Numerator = b;
            elseif nargin == 1
                obj.InterpolationFactor = a;
            elseif nargin == 4
                obj.InterpolationFactor = b;
                obj.Numerator = d;
            end
            matlab_dsp_init_state(obj);
        end
        function y = step(obj, x)
            y = matlab_dsp_firinterp_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end

classdef dsp_CICDecimator < handle
    properties
        DecimationFactor
        NumSections
        IntState
        CombState
        IsLocked
    end
    methods
        function obj = dsp_CICDecimator(a, b)
            obj.IsLocked = 0;
            obj.DecimationFactor = 4;
            obj.NumSections = 2;
            obj.IntState = 0;
            obj.CombState = 0;
            if nargin == 2
                obj.DecimationFactor = b;        % ('DecimationFactor', R)
            elseif nargin == 1
                obj.DecimationFactor = a;
            end
        end
        function y = step(obj, x)
            y = matlab_dsp_cicdecim_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end

classdef dsp_CICInterpolator < handle
    properties
        InterpolationFactor
        NumSections
        IntState
        CombState
        IsLocked
    end
    methods
        function obj = dsp_CICInterpolator(a, b)
            obj.IsLocked = 0;
            obj.InterpolationFactor = 4;
            obj.NumSections = 2;
            obj.IntState = 0;
            obj.CombState = 0;
            if nargin == 2
                obj.InterpolationFactor = b;
            elseif nargin == 1
                obj.InterpolationFactor = a;
            end
        end
        function y = step(obj, x)
            y = matlab_dsp_cicinterp_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end

classdef dsp_SampleRateConverter < handle
    properties
        InterpolationFactor
        DecimationFactor
        Numerator
        State
        IsLocked
    end
    methods
        function obj = dsp_SampleRateConverter(a, b, c)
            obj.IsLocked = 0;
            obj.InterpolationFactor = 1;
            obj.DecimationFactor = 1;
            obj.Numerator = 1;
            obj.State = zeros(1, 1);
            if nargin == 3
                obj.InterpolationFactor = a;     % (L, M, b) positional
                obj.DecimationFactor = b;
                obj.Numerator = c;
            elseif nargin == 2
                obj.InterpolationFactor = a;
                obj.DecimationFactor = b;
            end
            matlab_dsp_init_state(obj);
        end
        function y = step(obj, x)
            y = matlab_dsp_rateconv_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end

% ===================================================================
% Tier-5 — sources, streaming statistics, detectors, spectral
% ===================================================================

classdef dsp_SineWave < handle
    properties
        Frequency
        SampleRate
        Amplitude
        Phase
        SamplesPerFrame
        IsLocked
    end
    methods
        function obj = dsp_SineWave(a, b)
            obj.IsLocked = 0;
            obj.Frequency = 100;
            obj.SampleRate = 1000;
            obj.Amplitude = 1;
            obj.Phase = 0;
            obj.SamplesPerFrame = 64;
            if nargin == 1
                obj.Frequency = a;
            elseif nargin == 2
                obj.Frequency = b;        % ('Frequency', f)
            end
        end
        function y = step(obj)
            y = matlab_dsp_sine_step(obj);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
    end
end

classdef dsp_NCO < handle
    properties
        Frequency
        SampleRate
        Amplitude
        Phase
        SamplesPerFrame
        IsLocked
    end
    methods
        function obj = dsp_NCO(a, b)
            obj.IsLocked = 0;
            obj.Frequency = 100;
            obj.SampleRate = 1000;
            obj.Amplitude = 1;
            obj.Phase = 0;
            obj.SamplesPerFrame = 64;
            if nargin == 1
                obj.Frequency = a;
            elseif nargin == 2
                obj.Frequency = b;
            end
        end
        function y = step(obj)
            y = matlab_dsp_nco_step(obj);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
    end
end

classdef dsp_Chirp < handle
    properties
        InitialFrequency
        FrequencySweepRate
        SampleRate
        InstFreq
        Phase
        SamplesPerFrame
        IsLocked
    end
    methods
        function obj = dsp_Chirp(a, b)
            obj.IsLocked = 0;
            obj.InitialFrequency = 10;
            obj.FrequencySweepRate = 0;
            obj.SampleRate = 1000;
            obj.InstFreq = 0;
            obj.Phase = 0;
            obj.SamplesPerFrame = 64;
            if nargin == 2
                obj.InitialFrequency = a;
                obj.FrequencySweepRate = b;
            elseif nargin == 1
                obj.InitialFrequency = a;
            end
        end
        function y = step(obj)
            y = matlab_dsp_chirp_step(obj);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
            obj.InstFreq = 0;
            obj.Phase = 0;
        end
    end
end

% Single template for all moving-window statistics — only the step entry
% differs (mean / rms / max / min / std).  The constructor sets WindowLength
% (default 8) and lazily sizes Window in the runtime.
classdef dsp_MovingAverage < handle
    properties
        WindowLength
        Window
        WriteIdx
        IsLocked
    end
    methods
        function obj = dsp_MovingAverage(a, b)
            obj.IsLocked = 0;
            obj.WindowLength = 8;
            obj.Window = 0;
            obj.WriteIdx = 0;
            if nargin == 1
                obj.WindowLength = a;
            elseif nargin == 2
                obj.WindowLength = b;
            end
        end
        function y = step(obj, x)
            y = matlab_dsp_movavg_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
            obj.WriteIdx = 0;
        end
    end
end

classdef dsp_MovingRMS < handle
    properties
        WindowLength
        Window
        WriteIdx
        IsLocked
    end
    methods
        function obj = dsp_MovingRMS(a, b)
            obj.IsLocked = 0;
            obj.WindowLength = 8;
            obj.Window = 0;
            obj.WriteIdx = 0;
            if nargin == 1
                obj.WindowLength = a;
            elseif nargin == 2
                obj.WindowLength = b;
            end
        end
        function y = step(obj, x)
            y = matlab_dsp_movrms_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
            obj.WriteIdx = 0;
        end
    end
end

classdef dsp_MovingMaximum < handle
    properties
        WindowLength
        Window
        WriteIdx
        IsLocked
    end
    methods
        function obj = dsp_MovingMaximum(a, b)
            obj.IsLocked = 0;
            obj.WindowLength = 8;
            obj.Window = 0;
            obj.WriteIdx = 0;
            if nargin == 1
                obj.WindowLength = a;
            elseif nargin == 2
                obj.WindowLength = b;
            end
        end
        function y = step(obj, x)
            y = matlab_dsp_movmax_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
            obj.WriteIdx = 0;
        end
    end
end

classdef dsp_MovingStandardDeviation < handle
    properties
        WindowLength
        Window
        WriteIdx
        IsLocked
    end
    methods
        function obj = dsp_MovingStandardDeviation(a, b)
            obj.IsLocked = 0;
            obj.WindowLength = 8;
            obj.Window = 0;
            obj.WriteIdx = 0;
            if nargin == 1
                obj.WindowLength = a;
            elseif nargin == 2
                obj.WindowLength = b;
            end
        end
        function y = step(obj, x)
            y = matlab_dsp_movstd_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
            obj.WriteIdx = 0;
        end
    end
end

classdef dsp_PeakFinder < handle
    properties
        Prev
        Dir
        IsLocked
    end
    methods
        function obj = dsp_PeakFinder()
            obj.IsLocked = 0;
            obj.Prev = 0;
            obj.Dir = 0;
        end
        function y = step(obj, x)
            y = matlab_dsp_peakfind_step(obj, x);
        end
        function reset(obj)
            obj.Prev = 0;
            obj.Dir = 0;
        end
    end
end

classdef dsp_DCBlocker < handle
    properties
        Alpha
        Xprev
        Yprev
        IsLocked
    end
    methods
        function obj = dsp_DCBlocker(a)
            obj.IsLocked = 0;
            obj.Alpha = 0.995;
            obj.Xprev = 0;
            obj.Yprev = 0;
            if nargin == 1
                obj.Alpha = a;
            end
        end
        function y = step(obj, x)
            y = matlab_dsp_dcblock_step(obj, x);
        end
        function reset(obj)
            obj.Xprev = 0;
            obj.Yprev = 0;
        end
    end
end

classdef dsp_ZeroCrossingDetector < handle
    properties
        PrevSign
        IsLocked
    end
    methods
        function obj = dsp_ZeroCrossingDetector()
            obj.IsLocked = 0;
            obj.PrevSign = 0;
        end
        function y = step(obj, x)
            y = matlab_dsp_zcd_step(obj, x);
        end
        function reset(obj)
            obj.PrevSign = 0;
        end
    end
end

classdef dsp_SpectrumEstimator < handle
    properties
        FFTLength
        ForgettingFactor
        Buf
        Widx
        Filled
        PSD
        IsLocked
    end
    methods
        function obj = dsp_SpectrumEstimator(a, b)
            obj.IsLocked = 0;
            obj.FFTLength = 256;
            obj.ForgettingFactor = 0.9;
            obj.Buf = 0;
            obj.Widx = 0;
            obj.Filled = 0;
            obj.PSD = 0;
            if nargin == 2
                obj.FFTLength = b;       % ('FFTLength', K)
            elseif nargin == 1
                obj.FFTLength = a;
            end
        end
        function p = step(obj, x)
            p = matlab_dsp_spectest_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
            obj.Widx = 0;
            obj.Filled = 0;
        end
    end
end

classdef dsp_AsyncBuffer < handle
    properties
        Capacity
        Buf
        Widx
        Count
        IsLocked
    end
    methods
        function obj = dsp_AsyncBuffer(a, b)
            obj.IsLocked = 0;
            obj.Capacity = 1024;
            obj.Buf = 0;
            obj.Widx = 0;
            obj.Count = 0;
            if nargin == 1
                obj.Capacity = a;
            elseif nargin == 2
                obj.Capacity = b;
            end
        end
        function n = write(obj, x)
            n = matlab_dsp_asyncbuf_write(obj, x);
        end
        function y = read(obj, n)
            y = matlab_dsp_asyncbuf_read(obj, n);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
            obj.Widx = 0;
            obj.Count = 0;
        end
    end
end

% ===================================================================
% Tier-6 — linear-algebra SOs + polish filter SOs
% ===================================================================

classdef dsp_LevinsonSolver < handle
    properties
        IsLocked
    end
    methods
        function obj = dsp_LevinsonSolver()
            obj.IsLocked = 0;
        end
        function a = step(obj, r)
            a = matlab_dsp_levinson_step(obj, r);
        end
        function reset(obj)
            obj.IsLocked = 0;
        end
    end
end

classdef dsp_NotchPeakFilter < handle
    properties
        CenterFrequency
        Bandwidth
        IsPeak               % 0 = notch, 1 = peak
        State
        IsLocked
    end
    methods
        function obj = dsp_NotchPeakFilter(a, b)
            obj.IsLocked = 0;
            obj.CenterFrequency = 0.5;
            obj.Bandwidth = 0.1;
            obj.IsPeak = 0;
            obj.State = zeros(1, 2);
            if nargin == 1
                obj.CenterFrequency = a;
            elseif nargin == 2
                obj.CenterFrequency = a;
                obj.Bandwidth = b;
            end
        end
        function y = step(obj, x)
            y = matlab_dsp_notchpeak_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
    end
end

classdef dsp_LowpassFilter < handle
    properties
        FilterOrder
        CutoffFrequency
        Numerator
        State
        IsLocked
    end
    methods
        function obj = dsp_LowpassFilter(a, b)
            obj.IsLocked = 0;
            obj.FilterOrder = 32;
            obj.CutoffFrequency = 0.25;
            obj.Numerator = 0;          % lazily designed on first step
            obj.State = zeros(1, 1);
            if nargin == 1
                obj.CutoffFrequency = a;
            elseif nargin == 2
                obj.CutoffFrequency = b;     % ('CutoffFrequency', fc)
            end
        end
        function y = step(obj, x)
            y = matlab_dsp_lowpass_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
    end
end

classdef dsp_HighpassFilter < handle
    properties
        FilterOrder
        CutoffFrequency
        Numerator
        State
        IsLocked
    end
    methods
        function obj = dsp_HighpassFilter(a, b)
            obj.IsLocked = 0;
            obj.FilterOrder = 32;
            obj.CutoffFrequency = 0.5;
            obj.Numerator = 0;
            obj.State = zeros(1, 1);
            if nargin == 1
                obj.CutoffFrequency = a;
            elseif nargin == 2
                obj.CutoffFrequency = b;
            end
        end
        function y = step(obj, x)
            y = matlab_dsp_highpass_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
    end
end

classdef dsp_Delay < handle
    properties
        Length
        State
        IsLocked
    end
    methods
        function obj = dsp_Delay(a, b)
            obj.IsLocked = 0;
            obj.Length = 1;
            obj.State = zeros(1, 1);
            if nargin == 2
                obj.Length = b;          % ('Length', D)
            elseif nargin == 1
                obj.Length = a;          % (D)
            end
            matlab_dsp_init_state(obj);
        end
        function y = step(obj, x)
            y = matlab_dsp_delay_step(obj, x);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
        function tf = isLocked(obj)
            tf = obj.IsLocked;
        end
    end
end
