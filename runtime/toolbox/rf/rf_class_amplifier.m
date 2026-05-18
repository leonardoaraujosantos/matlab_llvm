% RFCktAmplifier — RF circuit amplifier block.
%
% Holds the canonical amplifier triple (NF in dB, small-signal gain
% in dB, input-referred IP3 in dBm) plus the operating frequency.
% Cascade with other rfckt blocks via the function-form
% `rfbudgetFriis(gains_dB, nfs_dB, ip3_dBm, p_in_dBm, bw_Hz)` —
% pass per-block column vectors of NF / Gain / IP3 to compose the chain.
%
% v1 ships the scalar property surface; richer S-parameter / noise-data
% population from .amp file or Touchstone is a follow-on.

classdef RFCktAmplifier < handle
    properties
        NF_dB
        Gain_dB
        IP3_dBm
        Frequency_Hz
    end
    methods
        function obj = RFCktAmplifier(nf_db, gain_db, ip3_dbm)
            if nargin >= 1
                obj.NF_dB = nf_db;
            else
                obj.NF_dB = 3.0;
            end
            if nargin >= 2
                obj.Gain_dB = gain_db;
            else
                obj.Gain_dB = 20.0;
            end
            if nargin >= 3
                obj.IP3_dBm = ip3_dbm;
            else
                obj.IP3_dBm = 30.0;
            end
            obj.Frequency_Hz = 1.0e9;
        end

        function s = analyze(obj, freqs)
            % Synthesize the matched-port amplifier S-parameters at the
            % requested frequencies via the function-form helper.
            s = rfAnalyzeAmplifier(obj.Gain_dB, freqs, 50.0);
        end
    end
end
