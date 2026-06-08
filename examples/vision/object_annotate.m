% object_annotate.m — Computer Vision Toolbox Phase-B (Tier-3).
% ----------------------------------------------------------------------
% Detect distinct objects in a REAL photograph (hardware tools on a wooden
% bench) and annotate them.  The metal objects are darker than the bench, so
% complement + Otsu binarize + connected-component labelling segments them;
% regionprops gives each object's bounding box; selectStrongestBbox keeps the
% non-overlapping detections; insertShape draws the boxes.  Result image:
%   /tmp/cv_annotated.png — the photo with object bounding boxes drawn

I = imread('data/tools.png');                % real top-down photo of tools

% Segment foreground objects (dark metal on a light bench).
bw = imbinarize(imcomplement(I));            % objects -> foreground
se = strel('disk', 2);
bw = imclose(bw, se);                        % fill highlight gaps within objects
bw = bwareaopen(bw, 150);                    % drop small noise blobs
L  = bwlabel(bw);
fprintf('connected components: %d\n', max(max(L)));

% Per-object bounding boxes + areas; keep the strongest non-overlapping ones.
boxes  = regionprops(L, 'BoundingBox');      % N x 4 [x y w h]
areas  = regionprops(L, 'Area');             % N x 1 scores
strong = selectStrongestBbox(boxes, areas);
fprintf('objects kept after NMS: %d\n', size(strong,1));

% Image output: draw the detected boxes onto the photo.
annotated = insertShape(I, 'rectangle', strong);
imwrite(annotated, '/tmp/cv_annotated.png');
fprintf('wrote /tmp/cv_annotated.png (%dx%d)\n', size(annotated,1), size(annotated,2));

% Report bounding-box overlap statistics (Tier-3 utility).
iou = bboxOverlapRatio(strong, strong);
fprintf('self-overlap diagonal (should be 1): %.2f\n', iou(1,1));
