% Phase 1.2 — varargout multi-return unpacking. Three call shapes:
%   1. pure varargout (`function varargout = trio(n)`)
%   2. mixed declared + varargout (`function [first, varargout] = decompose(n)`)
%   3. caller asks for fewer outputs than provided (must still pick the
%      first N off the cell rather than crashing).

[a, b, c] = trio(7);
disp(a);
disp(b);
disp(c);

[head, x, y, z] = decompose(100);
disp(head);
disp(x);
disp(y);
disp(z);

[p, q] = decompose(50);
disp(p);
disp(q);

% Plain multi-return (no varargout) — also part of this slice; was
% previously broken (every LHS got the same value).
[s, d] = sumdiff(10, 3);
disp(s);
disp(d);

function varargout = trio(n)
  varargout{1} = n;
  varargout{2} = n * 2;
  varargout{3} = n * 3;
end

function [first, varargout] = decompose(n)
  first = n;
  varargout{1} = n + 1;
  varargout{2} = n + 2;
  varargout{3} = n + 3;
end

function [s, d] = sumdiff(a, b)
  s = a + b;
  d = a - b;
end
