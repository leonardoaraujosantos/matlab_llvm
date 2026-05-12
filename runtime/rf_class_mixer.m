% RFCktMixer — RF circuit mixer block.
%
% Frequency-translating block with conversion gain, NF, IP3, and the
% local-oscillator frequency.  Cascades into rfbudget the same way an
% amplifier does: NF / Gain / IP3 column entries; the LO frequency is
% additional metadata for the analysis caller.

classdef RFCktMixer < handle
    properties
        NF_dB
        ConversionGain_dB
        IP3_dBm
        LO_Frequency_Hz
        IF_Frequency_Hz
    end
    methods
        function obj = RFCktMixer(nf_db, gain_db, ip3_dbm, lo_hz)
            if nargin >= 1
                obj.NF_dB = nf_db;
            else
                obj.NF_dB = 8.0;
            end
            if nargin >= 2
                obj.ConversionGain_dB = gain_db;
            else
                obj.ConversionGain_dB = -8.0;
            end
            if nargin >= 3
                obj.IP3_dBm = ip3_dbm;
            else
                obj.IP3_dBm = 10.0;
            end
            if nargin >= 4
                obj.LO_Frequency_Hz = lo_hz;
            else
                obj.LO_Frequency_Hz = 1.0e9;
            end
            obj.IF_Frequency_Hz = 0.0;
        end

        function s = analyze(obj, freqs)
            % Mixer S-parameters via the same forward-gain template as
            % an amplifier, with the conversion gain in place of small-
            % signal gain.
            s = rfAnalyzeAmplifier(obj.ConversionGain_dB, freqs, 50.0);
        end
    end
end
