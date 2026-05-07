% Tier-1 (Signal Processing Toolbox roadmap §2.4): partial-fraction
% expansion via residue. Distinct-pole scope; repeated-pole multiplicity
% grouping is a separate slice. Hand-checkable identities throughout.
%
% Order of (r, p) is solver-dependent (Durand-Kerner / numpy.roots
% don't agree on root ordering), so the golden tests symmetric
% functions of the residues + poles only.

% H(s) = 1 / ((s - 1)(s - 2)) = -1/(s-1) + 1/(s-2).
%   sum(r) = 0, sum(p) = 3, k empty.
b = [1];
a = [1 -3 2];
[r, p, k] = residue(b, a);
disp(sum(real(r)));      % 0
disp(sum(real(p)));      % 3
disp(size(k, 2));        % 0 (deg(b) < deg(a))

% H(s) = (s^2 + 1) / (s - 1) = s + 1 + 2/(s - 1).
%   k = [1, 1], one pole at 1 with residue 2.
b2 = [1 0 1];
a2 = [1 -1];
[r2, p2, k2] = residue(b2, a2);
disp(sum(real(r2)));     % 2
disp(sum(real(p2)));     % 1
disp(k2);                % [1 1]

% H(s) = (3s + 4) / ((s + 2)(s + 3)) = -2/(s+2) + 5/(s+3).
%   At s = -2: numerator = -2; residue = -2 / (-2 + 3) = -2.
%   At s = -3: numerator = -5; residue = -5 / (-3 + 2) =  5.
%   sum(r) = 3, prod(r) = -10. sum(p) = -5, prod(p) = 6. k empty.
b3 = [3 4];
a3 = [1 5 6];
[r3, p3, k3] = residue(b3, a3);
disp(sum(real(r3)));     % 3
disp(sum(real(p3)));     % -5
disp(prod(real(r3)));    % -10
disp(prod(real(p3)));    % 6
disp(size(k3, 2));       % 0
