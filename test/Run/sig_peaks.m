% Tier-3 (Signal Processing Toolbox roadmap §4.3): findpeaks +
% scalar reductions. Strict-monotonic peak definition (no plateaus,
% no endpoints). MinPeak* options deferred to follow-on slice.

% Two peaks: 5 at idx 4, 7 at idx 7. (Position 9 has value 4 with
% neighbors 5 and 3 — not a peak since 5 > 4.)
x = [1 2 3 5 4 6 7 5 4 3];
pks = findpeaks(x);
disp(pks);

% 2-return form: [pks, locs] (1-based MATLAB index).
[p, lc] = findpeaks(x);
disp(p);
disp(lc);

% Scalar reductions.
disp(rms([3 4]));            % sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
disp(peak2peak([1 5 -3 7]));  % 7 - (-3) = 10
disp(peak2rms([1 1 1 1]));    % max=1, rms=1 → 1
disp(rssq([3 4]));            % sqrt(9+16) = 5
