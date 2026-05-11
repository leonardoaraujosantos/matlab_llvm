% antenna_patterns.m
% ====================================================================
% Sample the analytical directional antenna patterns shipped in
% PROP-Tier-3 §3.4.1. Useful for sanity-checking gain values before
% feeding them into coverage_grid_multi. We unroll the angle sweeps
% rather than `for az = vec` since the latter is not on the compiler's
% supported front for scalar loop induction.
% ====================================================================

% Common parameters
gain_peak  = 17.0;
bw_az      = 65.0;
bw_el      = 10.0;
fb_dB      = 25.0;

disp('=== Sector pattern (gain dBi at sampled (az, el)) ===');
for k = 1:7
    if k == 1; az = -90.0;
    elseif k == 2; az = -45.0;
    elseif k == 3; az = -10.0;
    elseif k == 4; az =   0.0;
    elseif k == 5; az =  10.0;
    elseif k == 6; az =  45.0;
    else;          az =  90.0;
    end
    for j = 1:5
        if j == 1; el = -10.0;
        elseif j == 2; el = -5.0;
        elseif j == 3; el =  0.0;
        elseif j == 4; el =  5.0;
        else;          el = 10.0;
        end
        g = sectorPattern(az, el, bw_az, bw_el, gain_peak, fb_dB);
        fprintf('  az = %+5.1f deg, el = %+5.1f deg : %.2f dBi\n', az, el, g);
    end
end

disp(' ');
disp('=== Cosine-power pattern (dish, n = 30) ===');
% 22 dBi peak, half-bw 8 deg -> tight pencil beam.
for k = 1:7
    if k == 1; az = -20.0;
    elseif k == 2; az = -10.0;
    elseif k == 3; az =  -4.0;
    elseif k == 4; az =   0.0;
    elseif k == 5; az =   4.0;
    elseif k == 6; az =  10.0;
    else;          az =  20.0;
    end
    g = cosinePattern(az, 0, 8, 8, 22, 30);
    fprintf('  az = %+5.1f deg : %.2f dBi\n', az, g);
end

disp(' ');
disp('=== Gaussian pattern (no sidelobes, 30 deg half-bw) ===');
for k = 1:7
    if k == 1; az = -40.0;
    elseif k == 2; az = -20.0;
    elseif k == 3; az = -10.0;
    elseif k == 4; az =   0.0;
    elseif k == 5; az =  10.0;
    elseif k == 6; az =  20.0;
    else;          az =  40.0;
    end
    g = gaussianPattern(az, 0, 30, 10, 12);
    fprintf('  az = %+5.1f deg : %.2f dBi\n', az, g);
end

disp(' ');
disp('=== Mount orientation: world az = 90 deg, antenna pointing 60 deg ===');
az_local = applyMountAz(90.0, 0.0, 60.0, 0.0);
el_local = applyMountEl(90.0, 0.0, 60.0, 5.0);   % 5 deg electrical down-tilt
fprintf('  Local az (observer - boresight) : %.1f deg\n', az_local);
fprintf('  Local el (after 5 deg downtilt) : %.1f deg\n', el_local);
g_world = sectorPattern(az_local, el_local, 65, 10, 17, 25);
fprintf('  Apparent gain at observer       : %.2f dBi\n', g_world);
