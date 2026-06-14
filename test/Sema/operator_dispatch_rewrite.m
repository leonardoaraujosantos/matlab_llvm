% operator_dispatch_rewrite.m — #191 P3. With the class on the dispatch-desynth
% allow-list (Vec2), operator overloads are rewritten at Sema time into explicit
% method calls: `a + b` -> `a.plus(b)`, `a * 3` -> `a.mtimes(3)` (the scalar is
% NOT boxed for a non-box-safe class), and the unary `-a` -> `a.uminus()`. The
% dump below shows `Field .plus` / `Field .mtimes` / `Field .uminus`
% FieldAccess-callee nodes where BinaryOp / UnaryOp used to be — proof the
% rewrite fired (it lowers identically to the previously-synthesized
% Vec2__plus / Vec2__mtimes / Vec2__uminus). A class NOT on the allow-list
% keeps its BinaryOp / UnaryOp.

a = Vec2(1, 2);
b = Vec2(3, 4);
c = a + b;
s = a * 3;
d = -a;       % unary minus -> a.uminus() (#191 P3 uminus desynth)

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
    function r = uminus(a)
      r = Vec2(-a.x, -a.y);
    end
  end
end
