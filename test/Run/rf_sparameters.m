% RFSparameters catalog smoke test — scalar properties only.
%
% v1 ships the catalog skeleton; Touchstone reader + 3-D complex
% Parameters cube land in a follow-on slice.

p = RFSparameters(2, 50.0);
disp(p.NumPorts);          % 2
disp(p.Impedance);          % 50

% Default-constructed sparameters.
q = RFSparameters();
disp(q.NumPorts);          % 2 (default)
disp(q.Impedance);          % 50 (default)
