% funcstyle_method_dispatch.m — #191 P3 prerequisite. Function-style instance-
% method dispatch `meth(obj, ...)` recovers the method's inferred output type
% (the NameExpr-callee analog of P1.1's obj.method()), so the upcoming P3
% operator rewrite (`a*b` -> `mtimes(a,b)`) keeps the object<Class> result type
% instead of dropping it to none/Any. Uses `mtimes` (a method name that also
% resolves as a builtin, as the operator-method names do). A generic builtin
% that is NOT a method of the class is unaffected.

p = Pt(3);
q = mtimes(p, p);   % object<Pt>  (function-style method dispatch)
r = p.mtimes(p);    % object<Pt>  (FieldAccess form, P1.1 — same result)

classdef Pt
  properties
    v
  end
  methods
    function obj = Pt(x)
      obj.v = x;
    end
    function out = mtimes(a, b)
      out = Pt(a.v * b.v);
    end
  end
end
