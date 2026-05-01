% Phase 3 — value-class copy-on-assign semantics. A class without
% `< handle` defaults to value semantics in MATLAB; mutations to a
% copy must not leak back into the original.

a = Counter();
a.value = 10;
b = a;                  % clones
b.value = 99;           % must not change a
disp(a.value);          % 10
disp(b.value);          % 99

% Three-way: a fresh source can be copied to multiple bindings,
% each with independent state.
c = Counter();
c.value = 5;
d = c;
e = c;
e.value = 50;
disp(c.value);          % 5
disp(d.value);          % 5
disp(e.value);          % 50

classdef Counter
    properties
        value
    end
end
