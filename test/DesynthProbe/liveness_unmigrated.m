% #191 P3 — liveness fixture for the dispatch-desynth probe contract.
%
% `Gadget` is deliberately NOT on the desynth allow-list, so its operator
% overload is left to the lowering synthesis fallback and the
% MATLAB_LLVM_PROBE_LATE_MONO probe fires for it. This keeps the probe wiring
% under test even though every MIGRATED class (tf/ss/zpk/pid/frd/Vec2) is now
% fully desynthed (obj-LHS and scalar-LHS alike) and emits nothing.
a = Gadget(3);
b = Gadget(4);
c = a + b;
disp(c.v);

classdef Gadget
  properties
    v
  end
  methods
    function obj = Gadget(x)
      obj.v = x;
    end
    function r = plus(a, b)
      r = Gadget(a.v + b.v);
    end
  end
end
