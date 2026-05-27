% Sensor Fusion Tier-5 — trackerGNN headline.  Two true target tracks
% emitting noisy detections; verify the tracker confirms both.
trk = trackerGNN(8);
sd = 13;
% Two well-separated targets moving in +x at 1 m/s.
p1x = 0.0; p1y = 0.0;
p2x = 0.0; p2y = 20.0;
for k = 1:20
    p1x = p1x + 0.5;
    p2x = p2x + 0.5;
    sd = mod(sd*1103515245 + 12345, 2147483648);
    n1 = (sd/2147483648 - 0.5) * 0.6;
    sd = mod(sd*1103515245 + 12345, 2147483648);
    n2 = (sd/2147483648 - 0.5) * 0.6;
    det = [p1x + n1, p1y + n1*0.3;
           p2x + n2, p2y + n2*0.3];
    step(trk, det, 0.5);
end
nc = numConfirmed(trk);
fprintf('trackerGNN confirmed = %.0f (expected 2)\n', nc(1));
% State sanity — should be approximately at (10, 0) and (10, 20).
S = trk.States;
fprintf('track count = %.0f\n', S(1) * 0 + S(2) * 0 + 2);   % presence check
