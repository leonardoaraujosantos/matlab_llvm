% method_result_typing.m — #191 P1.1. A method call obj.method(args) now
% recovers the method's inferred first-output type (instead of Any): a method
% returning a constructor result types the call as object<Class>; a method
% with a param-independent output propagates that type. A method whose output
% depends on an untyped property still degrades to Any (that's the
% monomorphizer's job, P2.2). Builds on P2.1 (method bodies inferred first).

p = Pt(3);
q = p.scale(2);   % object<Pt>  (scale returns Pt(...))
d = p.dims();     % double      (param-independent constant output)
n = p.getv();     % any         (output depends on the untyped property v)

classdef Pt
  properties
    v
  end
  methods
    function obj = Pt(x)
      obj.v = x;
    end
    function r = scale(obj, k)
      r = Pt(obj.v * k);
    end
    function out = getv(obj)
      out = obj.v + 0;
    end
    function dd = dims(obj)
      dd = 2;
    end
  end
end
