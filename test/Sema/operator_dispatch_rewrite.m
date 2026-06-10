% operator_dispatch_rewrite.m — #191 P3. With the class on the dispatch-desynth
% allow-list (Vec2), operator overloads are rewritten at Sema time into explicit
% method calls: `a + b` -> `a.plus(b)`, `a * 3` -> `a.mtimes(3)` (the scalar is
% NOT boxed for a non-box-safe class). The dump below shows `Field .plus` /
% `Field .mtimes` FieldAccess-callee nodes where BinaryOps used to be — proof
% the rewrite fired (it lowers identically to the previously-synthesized
% Vec2__plus / Vec2__mtimes). A class NOT on the allow-list keeps its BinaryOp.

a = Vec2(1, 2);
b = Vec2(3, 4);
c = a + b;
s = a * 3;

classdef Vec2
  properties
    x
    y
  end
  methods
    function obj = Vec2(xv, yv)
      obj.x = xv;
      obj.y = yv;
    end
    function r = plus(a, b)
      r = Vec2(a.x + b.x, a.y + b.y);
    end
    function r = mtimes(a, k)
      r = Vec2(a.x * k, a.y * k);
    end
  end
end
