% Non-matched termination snp2smp via Schur complement.
%
% With matched terminations (z_term = z0), snp2smpZ should reduce to
% the simple sub-block extraction of snp2smp (matched case).  Use the
% 4-port fixture diff_pair.s4p and verify both forms agree for
% z_term = [50; 50] at the dropped ports.

data = touchstoneRead("fixtures/rf/diff_pair.s4p");
ports_kept = [1; 2];
z_term = [50.0; 50.0];   % matched terminations on dropped ports {3, 4}.

% Matched form (existing).
sub_match = snp2smp(data, ports_kept, 2);
disp(tsS11(sub_match));   % 0.1 / 0.15

% Non-matched form with z_term = z0 should agree.
sub_z = snp2smpZ(data, ports_kept, z_term, 2);
disp(tsS11(sub_z));        % same: 0.1 / 0.15

% Now apply non-matched terminations.  z_term = [100; 100] at the
% dropped ports — the Schur update kicks in and the kept-port S
% changes.
z_off = [100.0; 100.0];
sub_off = snp2smpZ(data, ports_kept, z_off, 2);
disp(tsS11(sub_off));      % differs from the matched value
