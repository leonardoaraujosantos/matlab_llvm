% optical_flow_motion.m — Computer Vision Toolbox Phase-B (Tier-4).
% ----------------------------------------------------------------------
% Estimate motion between two video frames with optical flow, on a REAL
% photograph.  The facade moves by a known vector between frames;
% opticalFlowLK (Lucas-Kanade) and opticalFlowHS (Horn-Schunck) recover the
% dominant flow.  The flow field is returned as [Vx; Vy] stacked vertically
% (2M x N).  Result image:
%   /tmp/cv_flow.png — per-pixel flow magnitude (bright = more motion)

frame1 = imread('data/facade.png');     % real frame
frame2 = imtranslate(frame1, [1 1]);    % camera pans +1 in x, +1 in y
M = size(frame1, 1);

flow = opticalFlowLK(frame1, frame2);
Vx = flow(1:M, :);
Vy = flow(M+1:2*M, :);

% On a real photo, flow is only well-conditioned where the image has texture
% (window/brick edges), so weight each estimate by the local edge strength to
% read off the dominant motion (a plain mean is diluted by flat regions).
hx = [1 0 -1; 2 0 -2; 1 0 -1];
hy = [1 2 1; 0 0 0; -1 -2 -1];
Gx = imfilter(frame1, hx);
Gy = imfilter(frame1, hy);
G  = Gx .* Gx + Gy .* Gy;               % gradient energy = edge-strength weight
wsum = sum(sum(G));
domVx = sum(sum(Vx .* G)) ./ wsum;
domVy = sum(sum(Vy .* G)) ./ wsum;
fprintf('Lucas-Kanade dominant flow: Vx=%.2f Vy=%.2f (true 1, 1)\n', domVx, domVy);

flowHS = opticalFlowHS(frame1, frame2);
VxHS = flowHS(1:M, :);
VyHS = flowHS(M+1:2*M, :);
hsVx = sum(sum(VxHS .* G)) ./ wsum;
hsVy = sum(sum(VyHS .* G)) ./ wsum;
fprintf('Horn-Schunck dominant flow: Vx=%.2f Vy=%.2f\n', hsVx, hsVy);

mag = sqrt(domVx .^ 2 + domVy .^ 2);
fprintf('dominant motion magnitude: %.2f px/frame\n', mag);

% Image output: per-pixel flow-energy map (bright = more motion).  Clamp to a
% small range so a few ill-conditioned pixels don't blow out the display.
mag2 = Vx .* Vx + Vy .* Vy;             % squared flow magnitude
flowImg = min(mag2, 4.0) ./ 4.0 .* 255;
imwrite(flowImg, '/tmp/cv_flow.png');
fprintf('wrote /tmp/cv_flow.png (%dx%d)\n', size(flowImg,1), size(flowImg,2));
