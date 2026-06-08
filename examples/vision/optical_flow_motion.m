% optical_flow_motion.m — Computer Vision Toolbox Phase-B (Tier-4).
% ----------------------------------------------------------------------
% Estimate motion between two video frames with optical flow.  A textured
% scene moves by a known vector; opticalFlowLK (Lucas-Kanade) and
% opticalFlowHS (Horn-Schunck) recover the dominant flow direction.  The flow
% field is returned as [Vx; Vy] stacked vertically (2M x N) over the shipped
% image-gradient substrate.

rng(5);
frame1 = imgaussfilt(rand(64, 64) * 255, 3);
frame2 = imtranslate(frame1, [2 1]);     % object moves +2 in x, +1 in y
M = size(frame1, 1);

flow = opticalFlowLK(frame1, frame2);
Vx = flow(1:M, :);
Vy = flow(M+1:2*M, :);
fprintf('Lucas-Kanade mean flow: Vx=%.2f Vy=%.2f (true 2, 1)\n', ...
        mean(mean(Vx)), mean(mean(Vy)));

flowHS = opticalFlowHS(frame1, frame2);
VxHS = flowHS(1:M, :);
VyHS = flowHS(M+1:2*M, :);
fprintf('Horn-Schunck mean flow: Vx=%.2f Vy=%.2f\n', ...
        mean(mean(VxHS)), mean(mean(VyHS)));

% dominant motion magnitude
mag = sqrt(mean(mean(Vx))^2 + mean(mean(Vy))^2);
fprintf('dominant motion magnitude: %.2f px/frame\n', mag);
