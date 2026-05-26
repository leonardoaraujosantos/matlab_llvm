% Financial Toolbox Tier-4 §2 — transition probabilities + CDS.

% --- transprob: cohort transition counts -> probability matrix. ---
% 3 rating states (A, B, Default). Counts row i -> col j.
counts = [ 90  8  2
            5 80 15
            0  0 100 ];
P = transprob(counts);
fprintf('P(1,:) = %.4f %.4f %.4f\n', P(1,1), P(1,2), P(1,3));   % .90 .08 .02
fprintf('P(2,:) = %.4f %.4f %.4f\n', P(2,1), P(2,2), P(2,3));   % .05 .80 .15
fprintf('P(3,:) = %.4f %.4f %.4f\n', P(3,1), P(3,2), P(3,3));   % 0 0 1 (absorbing)

% --- CDS bootstrap: flat 2% risk-free, rising CDS spreads, R=40%. ---
zr  = [0.02; 0.02; 0.02; 0.02; 0.02];
spr = [0.010; 0.012; 0.014; 0.016; 0.018];   % 100..180 bp
t   = [1; 2; 3; 4; 5];
surv = cdsbootstrap(zr, spr, t, 0.40);
fprintf('survival(1y) = %.4f\n', surv(1));
fprintf('survival(5y) = %.4f\n', surv(5));
% Survival should decrease monotonically; print the 1y-5y gap (>0).
fprintf('surv drop 1y->5y = %.4f\n', surv(1) - surv(5));

% --- Credit triangle: spread from hazard. ---
% hazard 3%, recovery 40% -> spread = 0.03 * 0.6 = 0.018 (180 bp).
fprintf('cdsspread(3%%, 40%%) = %.4f\n', cdsspread(0.03, 0.40));

% --- CDS MTM: market 200bp vs contract 150bp, RPV01 = 4.5. ---
% MTM = (0.02 - 0.015) * 4.5 = 0.0225.
fprintf('cdsprice = %.4f\n', cdsprice(0.02, 0.015, 4.5));
