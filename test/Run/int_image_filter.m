% Phase 1.1.G — integration: 8-bit image-style buffer flowing through
% saturating arithmetic and a comparison + cast back to logical mask.
% Exercises the uint8 lane end-to-end across casts, arithmetic, mixed
% scalar coercion, and the logical f64-lane comparison result.

img = uint8([100 150 200; 250 50 90; 30 200 255]);
disp(img);

% Brighten by 60 — the [200], [250], [200], [255] cells saturate at 255.
bright = img + 60;
disp(bright);

% Threshold mask via comparison (returns logical / f64 0-1).
mask = bright > 200;
disp(mask);

% Element-wise difference using the masked region.
diff = bright - img;
disp(diff);

% u8 -> i32 widening: same value, wider lane.
wide = int32(img);
disp(wide);
disp(wide + 1000000);    % proves we have int32 headroom now

