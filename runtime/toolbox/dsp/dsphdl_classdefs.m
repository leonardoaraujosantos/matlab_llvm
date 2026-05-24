% dsphdl_classdefs.m — DSP HDL Toolbox System-Object simulation surface.
%
% Tiers 7 / 8 of docs/dsp_toolbox_roadmap.md.  These classdefs are the
% MATLAB-side façade for the `dsphdl.*` hardware family (the cycle-
% accurate counterpart of `dsp.*`).  In simulation they compute the same
% reference result as their `dsp.*` siblings; the distinguishing parts —
% the synthesizable SystemVerilog emit with valid / backpressure-ready /
% reset ports and the cocotb SIL test — sit as a documented follow-on
% that depends on the emit-SV lane growing new patterns for clocked
% valid/ready datapaths.
%
% Each object exposes:
%   - A `Latency` property (the pipeline depth a real HW build would
%     introduce: FIR adder-tree depth + delay-line length, etc.).
%   - A `getLatency` method that returns it (programmatic introspection).
%   - The same `y = obj(x)` streaming step API as the corresponding
%     `dsp.*` object.  The valid/ready streaming control (`[y, v] =
%     obj(x, validIn)`) is carved to the SV-emit follow-on.

classdef dsphdl_FIRFilter < handle
    properties
        Numerator
        State
        Latency
        IsLocked
    end
    methods
        function obj = dsphdl_FIRFilter(a, b)
            obj.IsLocked = 0;
            obj.Numerator = 1;
            obj.State = zeros(1, 1);
            obj.Latency = 1;
            if nargin == 2
                obj.Numerator = b;
            elseif nargin == 1
                obj.Numerator = a;
            end
            matlab_dsp_init_state(obj);
        end
        function y = step(obj, x)
            y = matlab_dsphdl_fir_step(obj, x);
        end
        function L = getLatency(obj)
            L = matlab_dsphdl_latency(obj);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
    end
end

classdef dsphdl_BiquadFilter < handle
    properties
        SOSMatrix
        ScaleValues
        State
        Latency
        IsLocked
    end
    methods
        function obj = dsphdl_BiquadFilter(a, b)
            obj.IsLocked = 0;
            obj.ScaleValues = 1;
            obj.SOSMatrix = [1 0 0 1 0 0];
            obj.State = zeros(1, 2);
            obj.Latency = 3;
            if nargin == 2
                obj.SOSMatrix = b;
            elseif nargin == 1
                obj.SOSMatrix = a;
            end
            matlab_dsp_init_state(obj);
        end
        function y = step(obj, x)
            y = matlab_dsphdl_biquad_step(obj, x);
        end
        function L = getLatency(obj)
            L = matlab_dsphdl_latency(obj);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
    end
end

classdef dsphdl_SineWave < handle
    properties
        Frequency
        SampleRate
        Amplitude
        Phase
        SamplesPerFrame
        Latency
        IsLocked
    end
    methods
        function obj = dsphdl_SineWave(a, b)
            obj.IsLocked = 0;
            obj.Frequency = 100;
            obj.SampleRate = 1000;
            obj.Amplitude = 1;
            obj.Phase = 0;
            obj.SamplesPerFrame = 64;
            obj.Latency = 2;          % LUT lookup + register
            if nargin == 1
                obj.Frequency = a;
            elseif nargin == 2
                obj.Frequency = b;
            end
        end
        function y = step(obj)
            y = matlab_dsphdl_sine_step(obj);
        end
        function L = getLatency(obj)
            L = matlab_dsphdl_latency(obj);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
    end
end

classdef dsphdl_NCO < handle
    properties
        Frequency
        SampleRate
        Amplitude
        Phase
        SamplesPerFrame
        Latency
        IsLocked
    end
    methods
        function obj = dsphdl_NCO(a, b)
            obj.IsLocked = 0;
            obj.Frequency = 100;
            obj.SampleRate = 1000;
            obj.Amplitude = 1;
            obj.Phase = 0;
            obj.SamplesPerFrame = 64;
            obj.Latency = 4;          % accumulator + LUT + register
            if nargin == 1
                obj.Frequency = a;
            elseif nargin == 2
                obj.Frequency = b;
            end
        end
        function y = step(obj)
            y = matlab_dsphdl_nco_step(obj);
        end
        function L = getLatency(obj)
            L = matlab_dsphdl_latency(obj);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
    end
end

classdef dsphdl_FIRDecimator < handle
    properties
        DecimationFactor
        Numerator
        State
        Latency
        IsLocked
    end
    methods
        function obj = dsphdl_FIRDecimator(a, b, c, d)
            obj.IsLocked = 0;
            obj.DecimationFactor = 2;
            obj.Numerator = 1;
            obj.State = zeros(1, 1);
            obj.Latency = 5;
            if nargin == 2
                obj.DecimationFactor = a;
                obj.Numerator = b;
            elseif nargin == 1
                obj.DecimationFactor = a;
            elseif nargin == 4
                obj.DecimationFactor = b;
                obj.Numerator = d;
            end
            matlab_dsp_init_state(obj);
        end
        function y = step(obj, x)
            y = matlab_dsphdl_firdecim_step(obj, x);
        end
        function L = getLatency(obj)
            L = matlab_dsphdl_latency(obj);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
    end
end

classdef dsphdl_CICDecimator < handle
    properties
        DecimationFactor
        NumSections
        IntState
        CombState
        Latency
        IsLocked
    end
    methods
        function obj = dsphdl_CICDecimator(a, b)
            obj.IsLocked = 0;
            obj.DecimationFactor = 4;
            obj.NumSections = 2;
            obj.IntState = 0;
            obj.CombState = 0;
            obj.Latency = 8;
            if nargin == 2
                obj.DecimationFactor = b;
            elseif nargin == 1
                obj.DecimationFactor = a;
            end
        end
        function y = step(obj, x)
            y = matlab_dsphdl_cicdecim_step(obj, x);
        end
        function L = getLatency(obj)
            L = matlab_dsphdl_latency(obj);
        end
        function reset(obj)
            matlab_dsp_reset(obj);
        end
    end
end
