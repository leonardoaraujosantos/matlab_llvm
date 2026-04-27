% Synthesis-typed call site for examples/hdl/vector_processor.m. Run with:
%
%   just emit-sv-multi examples/hdl/vector_processor_synth.m \
%                      examples/hdl/vector_processor.m
%
% The two `fi([...], T)` literal-init args feed Stage C's static
% array-literal lowering, which produces a typed `!llvm.ptr` at the
% call site. Stage B's vector-function-arg path (Sema infers vector
% shape from `vec_a(k)` subscripts; LowerStaticFiArrays' Stage B
% extension rewrites the body's runtime subscripts to GEP+load on
% the arg pointer; the SV emitter renders the arg as
% `input logic signed [15:0] vec_a [3]`) connects them.
T = numerictype(1, 16, 8);
[m, d] = vector_processor(fi([1, 2, 3], T), fi([4, 5, 6], T));
disp(m);
disp(d);
