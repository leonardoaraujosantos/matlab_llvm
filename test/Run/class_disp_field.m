% Regression test for `disp(obj.Property)` lowering across the
% emit-{c,cpp,python,typescript} lanes. The MLIR lowerer routes
% `disp` of a class-pinned field-access through the
% `matlab_obj_disp_field` runtime call so the property's stored
% kind picks the right disp variant at runtime. Without the
% per-target rewrite (EmitC's `matlab_obj_disp_field` substitution +
% runtime stubs in matlab_runtime.py / matlab_runtime.ts) the
% generated source either fails to compile (struct value passed
% where void* is expected) or fails at import (missing runtime
% symbol).

a = HoldsScalar(3.14);
disp(a.Value);               % 3.14

b = HoldsScalar(0);
b.Value = -7.5;
disp(b.Value);               % -7.5

c = HoldsTwo(1.5, 4.0);
disp(c.A);                   % 1.5
disp(c.B);                   % 4

classdef HoldsScalar
    properties
        Value
    end
    methods
        function obj = HoldsScalar(v)
            if nargin == 1
                obj.Value = v;
            end
        end
    end
end

classdef HoldsTwo
    properties
        A
        B
    end
    methods
        function obj = HoldsTwo(a, b)
            if nargin == 2
                obj.A = a;
                obj.B = b;
            end
        end
    end
end
