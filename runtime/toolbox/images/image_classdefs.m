% Image Processing Toolbox — Tier-3 classdef umbrella.
% Auto-prepended by matlabc when the user input mentions `affine2d`,
% `projective2d`, `imref2d`, or `fitgeotform2d`.  The geometric-transform
% objects are thin holders for the 3x3 forward matrix T (post-multiply
% convention: [x y 1] * T); all warping lives in
% runtime/toolbox/images/runtime_images.cpp.
%
% affine2d(M) / projective2d(M) and fitgeotform2d(...) are NOT classdef
% methods — they are dispatched in Lowering.cpp (the matrix constructor
% writes T; fitgeotform2d allocs an affine2d then calls the runtime
% populate step matlab_image_fitgeo_init).  imwarp(A, tform) reads T + Kind
% at runtime.  Kind: 1 = affine, 2 = projective.

classdef affine2d
    properties
        T matrix
        Kind
    end
    methods
        function obj = affine2d()
            obj.T    = [1 0 0; 0 1 0; 0 0 1];
            obj.Kind = 1;
        end
    end
end

classdef projective2d
    properties
        T matrix
        Kind
    end
    methods
        function obj = projective2d()
            obj.T    = [1 0 0; 0 1 0; 0 0 1];
            obj.Kind = 2;
        end
    end
end

classdef imref2d
    properties
        ImageSize matrix
        XWorldLimits matrix
        YWorldLimits matrix
    end
    methods
        function obj = imref2d()
            obj.ImageSize    = zeros(1, 2);
            obj.XWorldLimits = zeros(1, 2);
            obj.YWorldLimits = zeros(1, 2);
        end
    end
end
