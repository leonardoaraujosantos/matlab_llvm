% Computer Vision Toolbox Tier-3/4 — bounding boxes + annotation + optical flow.
%   bboxOverlapRatio / selectStrongestBbox / bbox2points / insertShape;
%   opticalFlowLK recovers a known image translation.

% --- Tier-3: bounding-box utilities ---
A = [10 10 20 20];
B = [10 10 20 20; 12 12 20 20; 100 100 10 10];
iou = bboxOverlapRatio(A, B);
fprintf('IoU x1000: %.0f %.0f %.0f\n', round(iou(1)*1000), round(iou(2)*1000), round(iou(3)*1000));

boxes  = [10 10 20 20; 12 12 20 20; 100 100 10 10];
scores = [0.9; 0.8; 0.95];
kept = selectStrongestBbox(boxes, scores);
fprintf('NMS kept boxes: %.0f\n', size(kept,1));    % 2 (overlapping pair merged)

c = bbox2points([10 20 30 40]);
fprintf('box corners TL=(%.0f,%.0f) BR=(%.0f,%.0f)\n', c(1,1), c(1,2), c(3,1), c(3,2));

I = checkerboard(8, 3, 3) * 255;
J = insertShape(I, 'rectangle', [5 5 20 20]);
fprintf('annotation drew (max=%.0f, same size %.0f)\n', round(max(max(J))), size(J,1)-size(I,1));

% --- Tier-4: optical flow recovers a translation ---
rng(3);
G  = imgaussfilt(rand(60,60)*255, 3);
G1 = imtranslate(G, [1 0]);
flow = opticalFlowLK(G, G1);
H = size(G,1);
Vx = flow(1:H, :);
Vy = flow(H+1:2*H, :);
fprintf('LK flow: Vx=%.0f Vy=%.0f (expect 1, 0)\n', round(mean(mean(Vx))), abs(round(mean(mean(Vy)))));
flowHS = opticalFlowHS(G, G1);
fprintf('HS flow size %.0fx%.0f\n', size(flowHS,1), size(flowHS,2));   % 120 x 60
