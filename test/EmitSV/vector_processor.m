% Phase 5.6 Stage B SV — vector function arguments.
%
% `function [...] = f(vec_a, vec_b)` with vector-typed parameters
% lowers end-to-end:
%
%   Sema TypeInference scans the body for `param(k)` constant-
%   subscript sites. The largest k seen seeds the parameter's
%   inferred type as `Array{Unknown, Vector(N)}`; the body's
%   `vec_a = fi(vec_a, S, W, F)` re-cast then refines the element
%   to fi(spec). Without that pre-pass, the param defaults to
%   scalar i16 and `vec_a(k)` surfaces as `subscript(scalar, idx)` —
%   malformed.
%
%   Lowering attaches `matlab.array_n` / `matlab.fi_*` arg attrs
%   so downstream passes can reattach the shape that TypeMapper
%   loses by collapsing fi-array → !llvm.ptr.
%
%   LowerStaticFiArrays Stage B extension rewrites
%   `matlab_mat_i64_subscript1_s(arg_load, idx)` calls in the
%   body to GEP+load on the arg pointer. The no-op
%   `matlab.fi.cast(load_arg) → ptr` from `vec_a = fi(vec_a, ...)`
%   is dropped (it's identity for an already-typed vector arg).
%
%   HWLegalize accepts !llvm.ptr params with the `matlab.array_n`
%   metadata. The SV emitter renders them as
%   `input logic signed [W-1:0] vec_a [N]`.
T = numerictype(1, 16, 8);
[m, d] = vector_processor(fi([1, 2, 3], T), fi([4, 5, 6], T));
disp(m);
disp(d);

function [mag_sq, dot_prod] = vector_processor(vec_a, vec_b)
    %#codegen
    vec_a = fi(vec_a, 1, 16, 8);
    vec_b = fi(vec_b, 1, 16, 8);

    p1 = vec_a(1) * vec_b(1);
    p2 = vec_a(2) * vec_b(2);
    p3 = vec_a(3) * vec_b(3);

    dot_prod = p1 + p2 + p3;

    a1_sq = vec_a(1) * vec_a(1);
    a2_sq = vec_a(2) * vec_a(2);
    a3_sq = vec_a(3) * vec_a(3);

    mag_sq = a1_sq + a2_sq + a3_sq;
end
