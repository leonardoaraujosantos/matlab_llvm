% Tier-1 + infra: writeVerilogA on an RFRational classdef instance.
% Exercises:
%   - obj.A = matrix_value lowering via matlab_obj_set_mat (was
%     previously routed to obj_set_f64 because Rhs type was unresolved)
%   - matlab_struct_get_mat zero-size fall-back in writeVerilogA
%     (Poles missing -> read A from the classdef property)

K = 30;
ln10 = 2.302585092994046;
freqs = zeros(K, 1);
for k = 1:K
    t = (k - 1.0) / (K - 1.0);
    freqs(k) = exp((7.0 + t * 3.0) * ln10);
end

h_re = zeros(K, 1);
h_im = zeros(K, 1);
for k = 1:K
    w = 2.0 * 3.141592653589793 * freqs(k);
    a1 = 1.0e9;  c1 = 5.0e9;
    a2 = 3.0e9;  c2 = 1.0e10;
    den1 = a1*a1 + w*w;
    den2 = a2*a2 + w*w;
    h_re(k) = c1 * a1 / den1 + c2 * a2 / den2 + 0.25;
    h_im(k) = -c1 * w / den1 - c2 * w / den2;
end

mdl_struct = rationalfit(freqs, h_re, h_im, 2, 15);

obj       = RFRational();
obj.A     = rfPoles(mdl_struct);
obj.C     = rfResidues(mdl_struct);
obj.D     = rfD(mdl_struct);
obj.Delay = 1.0e-9;
obj.Order = rfOrder(mdl_struct);
obj.Error = rfFitError(mdl_struct);

ok = writeVerilogA(obj, "/tmp/_rf_writeva_classdef.va");
disp(ok);              % 1
