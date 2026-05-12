% RFRational — value classdef wrapping the rationalfit result.
%
% MathWorks-API parallel of rfmodel.rational with the same property
% surface: A (poles), C (residues), D (direct term), Delay (bulk
% delay), Order, Error.  Population is typically from the rationalfit
% return struct via field assignment:
%
%   mdl_struct = rationalfit(freq, h_re, h_im, n_poles, n_iter);
%   mdl = RFRational();
%   mdl.A = rfPoles(mdl_struct);
%   mdl.C = rfResidues(mdl_struct);
%   mdl.D = rfD(mdl_struct);
%   mdl.Order = rfOrder(mdl_struct);
%   mdl.Error = rfFitError(mdl_struct);
%
% Once shipped, `freqresp(mdl, freqs)` / `timeresp(mdl, u, ts)` /
% `passivity(mdl, ...)` accept either an RFRational instance or the
% raw struct interchangeably (handled by the runtime via the layout-
% magic-aware getters).

classdef RFRational < handle
    properties
        A           % poles (complex column)
        C           % residues (complex column)
        D           % direct term (real scalar)
        Delay       % bulk delay (real scalar, seconds)
        Order       % number of poles
        Error       % L2 relative fit error
    end
    methods
        function obj = RFRational()
            obj.D = 0.0;
            obj.Delay = 0.0;
            obj.Order = 0;
            obj.Error = 0.0;
        end
    end
end
