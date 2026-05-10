% tf model object — §3.1 minimum-viable surface (slice 1).
%
% What's shipped here:
%   - tf(num, den) constructor stores the polynomial vectors.
%   - obj.Numerator / obj.Denominator read access (CST classes
%     route to matlab_obj_get_mat regardless of the Sema-inferred
%     scalar default — see lib/MLIR/Lowering.cpp:6926).
%
% What's NOT shipped (follow-on slice — see
% docs/control_toolbox_roadmap.md §12):
%   - tf-vs-tf operator overloads (`G + H`, `G * H`). The lowering
%     path works in isolation but multi-call-site monomorphization
%     creates duplicate slot allocs in the cloned constructor body.
%   - Scalar mixing (`s + 2`, `tf('s')` builder). Needs an `isa`
%     runtime + a Sema-level CST-class property type tracker so
%     `obj.Field` reads carry the right type into downstream
%     arithmetic.

G = tf([1 2], [1 3 5]);
disp(G.Numerator);
disp(G.Denominator);
