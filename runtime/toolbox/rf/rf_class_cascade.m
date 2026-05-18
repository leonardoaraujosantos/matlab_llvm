% RFCktCascade — series cascade of RF circuit blocks.
%
% Holds the per-block NF / Gain / IP3 column vectors that compose the
% chain.  The user assembles the columns from a list of RFCktAmplifier
% / RFCktMixer / RFCktPassive blocks (one entry per block).  The
% `analyze(chain, p_in_dBm, bw_Hz)` method delegates to the function-
% form `rfbudgetFriis` which returns the cascaded gain / NF / IP3 /
% SNR / output power.
%
% v1 ships the property surface; helpers that build the columns from
% a heterogeneous block list and a `chain.append(block)` extension API
% are follow-on polish.

classdef RFCktCascade < handle
    properties
        Gains_dB        % column vector of per-block gain (dB)
        NFs_dB          % column vector of per-block NF (dB)
        IP3s_dBm        % column vector of per-block input IP3 (dBm)
        Frequency_Hz
        Label
    end
    methods
        function obj = RFCktCascade(gains_db, nfs_db, ip3_dbm)
            if nargin >= 1
                obj.Gains_dB = gains_db;
            end
            if nargin >= 2
                obj.NFs_dB = nfs_db;
            end
            if nargin >= 3
                obj.IP3s_dBm = ip3_dbm;
            end
            obj.Frequency_Hz = 1.0e9;
            obj.Label = "cascade";
        end
    end
end
