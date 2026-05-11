% Antenna Toolbox catalog smoke test — AntDipole + AntMonopole.
%
% ANT-Tier-1 ships geometry-only catalog classdefs; pattern /
% impedance / sparameters methods require the ANT-Tier-2 wire-MoM
% solver and land in a follow-on slice.  This test exercises just
% the construction + property reads.

d = AntDipole(2.0, 0.05, 0.0);
disp(d.Length);              % 2
disp(d.Width);                % 0.05
disp(d.FeedOffset);           % 0

% Default-constructed dipole.
d2 = AntDipole();
disp(d2.Length);              % 2 (default)
disp(d2.Width);                % 0.05 (default)

m = AntMonopole(1.0, 0.05, 2.0);
disp(m.Height);               % 1
disp(m.GroundPlaneLength);    % 2
