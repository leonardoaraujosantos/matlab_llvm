% Phase 4.5.1 SV — 6-operation ALU exercising multi-store scalar
% slot retyping. The user writes `data_out = fi(0, ...)` once and
% then conditionally overwrites it from each `case` arm; the slot
% stayed `none`-typed under the original pipeline because user-call
% refinement didn't see all the multi-block stores. The Phase 4.5.1
% RefineSlotTypes pass picks every `matlab.alloc` whose every store
% agrees on a concrete scalar primitive and retypes the slot to it.
%
% Also exercises Phase 4.5.3 (`overflow = false` → `arith.constant
% 0 : i1` → `output logic y1`) and the Phase 4 case-cascade
% lowering of `switch sel`.
T = numerictype(1, 16, 0);
S = numerictype(0, 8, 0);
[d, o] = alu_16bit(fi(5, T), fi(3, T), fi(2, S));
disp(d);

function [data_out, overflow] = alu_16bit(a, b, sel)
    %#codegen
    data_out = fi(0, 1, 16, 0);
    overflow = false;
    switch sel
        case 0
            data_out = a + b;
        case 1
            data_out = a - b;
        case 2
            data_out = bitand(a, b);
        case 3
            data_out = bitor(a, b);
        case 4
            data_out = bitxor(a, b);
        case 5
            data_out = bitcmp(a);
        otherwise
            data_out = fi(0, 1, 16, 0);
    end
end
