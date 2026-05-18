% RFCktPassive — generic passive 2-port (no gain, no NF beyond loss).
%
% Holds the small-signal insertion loss (dB) and a label.  Passive
% blocks cascade through rfbudget with Gain = -loss, NF = loss (for a
% lossy 2-port at room temperature, NF equals the loss), and a very
% high IP3 (well above any drive level the chain will see).

classdef RFCktPassive < handle
    properties
        Loss_dB
        Frequency_Hz
        Label
    end
    methods
        function obj = RFCktPassive(loss_db)
            if nargin >= 1
                obj.Loss_dB = loss_db;
            else
                obj.Loss_dB = 0.5;
            end
            obj.Frequency_Hz = 1.0e9;
            obj.Label = "passive";
        end

        function s = analyze(obj, freqs)
            s = rfAnalyzePassive(obj.Loss_dB, freqs, 50.0);
        end
    end
end
