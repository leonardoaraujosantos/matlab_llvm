% Sensor Fusion and Tracking Toolbox — classdef umbrella (Tiers 1–3).
% Auto-prepended by matlabc when the user input mentions any fusion symbol
% (`quaternion`, `trackingKF`, `trackingEKF`, `trackingUKF`, `imuSensor`,
% `gpsSensor`, `ahrsfilter`, `imufilter`, `complementaryFilter`,
% `insfilterMARG`, `objectDetection`).
%
% Every classdef here ships a *neutral* zero-arg constructor only; the
% user-facing forms (e.g. `quaternion([w x y z])`, `trackingEKF(x0,P0,Q,R)`,
% `imuSensor('SampleRate',fs)`) are intercepted in Lowering.cpp's
% constructor path (same pattern as `extendedKalmanFilter` in Ident) and
% populated by a `matlab_fusion_*_init` runtime call.  Method bodies
% forward to `matlab_fusion_*` runtime symbols.

% Tier-1 ------------------------------------------------------------------

classdef quaternion
    properties
        Data matrix   % N×4 [w x y z] rows; the canonical storage.
    end
    methods
        function obj = quaternion()
            obj.Data = zeros(1, 4);
            obj.Data(1) = 1;  % identity rotation by default
        end
        function disp(obj)
            matlab_fusion_quat_disp(obj);
        end
        function p = parts(obj)
            p = matlab_fusion_quat_parts(obj);
        end
        function e = eulerd(obj)
            % MATLAB's eulerd returns degrees; convert from our radian core.
            e = (180/pi) * matlab_fusion_quat_to_eul(obj.Data);
        end
        function e = euler(obj)
            e = matlab_fusion_quat_to_eul(obj.Data);
        end
        function r = rotmat(obj)
            % Point-rotation convention by default.
            r = matlab_fusion_quat_to_rotm(obj.Data, 0);
        end
        function n = norm(obj)
            n = matlab_fusion_quat_norm_data(obj.Data);
        end
    end
end

% Tier-2 ------------------------------------------------------------------
%
% MotionModel discriminant (Model property):
%   1 = constvel (constant velocity)
%   2 = constacc (constant acceleration)
%   3 = constturn (coordinated-turn)
%   4 = singer (Singer acceleration)

classdef trackingKF
    properties
        State matrix              % nx × 1
        StateCovariance matrix    % nx × nx
        ProcessNoise matrix       % nx × nx
        MeasurementNoise matrix   % ny × ny
        Model                     % motion-model discriminant
        Hmat matrix               % ny × nx measurement matrix (set by ctor)
        Fmat matrix               % nx × nx state-transition matrix (set by ctor)
    end
    methods
        function obj = trackingKF()
            obj.State            = zeros(1, 1);
            obj.StateCovariance  = zeros(1, 1);
            obj.ProcessNoise     = zeros(1, 1);
            obj.MeasurementNoise = zeros(1, 1);
            obj.Model            = 1;
            obj.Hmat             = zeros(1, 1);
            obj.Fmat             = zeros(1, 1);
        end
    end
end

classdef trackingEKF
    properties
        State matrix
        StateCovariance matrix
        ProcessNoise matrix
        MeasurementNoise matrix
    end
    methods
        function obj = trackingEKF()
            obj.State            = zeros(1, 1);
            obj.StateCovariance  = zeros(1, 1);
            obj.ProcessNoise     = zeros(1, 1);
            obj.MeasurementNoise = zeros(1, 1);
        end
    end
end

classdef trackingUKF
    properties
        State matrix
        StateCovariance matrix
        ProcessNoise matrix
        MeasurementNoise matrix
    end
    methods
        function obj = trackingUKF()
            obj.State            = zeros(1, 1);
            obj.StateCovariance  = zeros(1, 1);
            obj.ProcessNoise     = zeros(1, 1);
            obj.MeasurementNoise = zeros(1, 1);
        end
    end
end

% objectDetection — the measurement container.
classdef objectDetection
    properties
        Time
        Measurement matrix
        MeasurementNoise matrix
        SensorIndex
        ObjectClassID
    end
    methods
        function obj = objectDetection()
            obj.Time             = 0;
            obj.Measurement      = zeros(0, 0);
            obj.MeasurementNoise = zeros(0, 0);
            obj.SensorIndex      = 1;
            obj.ObjectClassID    = 0;
        end
    end
end

% Tier-3 ------------------------------------------------------------------

classdef imuSensor
    properties
        SampleRate
        % Accelerometer noise parameters.
        AccelBias matrix
        AccelNoiseDensity matrix
        % Gyroscope noise parameters.
        GyroBias matrix
        GyroNoiseDensity matrix
        % Magnetometer noise parameters (used only if HasMagnetometer=1).
        MagBias matrix
        MagNoiseDensity matrix
        % Reference (navigation) frame fields, for sample generation:
        Gravity                   % m/s^2 (default 9.81)
        MagneticFieldNED matrix   % 1×3 µT
        HasMagnetometer
    end
    methods
        function obj = imuSensor()
            obj.SampleRate         = 100;
            obj.AccelBias          = zeros(1, 3);
            obj.AccelNoiseDensity  = 1e-3 * ones(1, 3);
            obj.GyroBias           = zeros(1, 3);
            obj.GyroNoiseDensity   = 1e-4 * ones(1, 3);
            obj.MagBias            = zeros(1, 3);
            obj.MagNoiseDensity    = 1e-1 * ones(1, 3);
            obj.Gravity            = 9.81;
            obj.MagneticFieldNED   = [27.555, -2.4169, -16.0849];  % µT
            obj.HasMagnetometer    = 0;
        end
    end
end

classdef gpsSensor
    properties
        SampleRate
        PositionNoise matrix      % 1×3 σ (m) on lla/ned
        VelocityNoise matrix      % 1×3 σ (m/s)
        ReferenceLocation matrix  % 1×3 [lat lon alt]
    end
    methods
        function obj = gpsSensor()
            obj.SampleRate         = 1;
            obj.PositionNoise      = ones(1, 3);
            obj.VelocityNoise      = 0.1 * ones(1, 3);
            obj.ReferenceLocation  = zeros(1, 3);
        end
    end
end

% ahrsfilter — accel+gyro+mag orientation EKF.  Carries the quaternion
% orientation state via a 4-element [w x y z] vector inside State, plus the
% gyro-bias estimate.
classdef ahrsfilter
    properties
        SampleRate
        State matrix              % [q_w q_x q_y q_z gx_b gy_b gz_b] (7×1)
        StateCovariance matrix    % 7×7
        % Tuning parameters (single scalar gains, matching MATLAB defaults).
        AccelerometerNoise
        MagnetometerNoise
        GyroscopeNoise
        GyroscopeDriftNoise
        LinearAccelerationNoise
        MagneticDisturbanceNoise
        ExpectedMagneticFieldStrength
    end
    methods
        function obj = ahrsfilter()
            obj.SampleRate                    = 100;
            obj.State                         = zeros(7, 1);
            obj.State(1)                      = 1;       % q = identity
            obj.StateCovariance               = 1e-3 * eye(7);
            obj.AccelerometerNoise            = 1e-4;
            obj.MagnetometerNoise             = 1e-3;
            obj.GyroscopeNoise                = 1e-6;
            obj.GyroscopeDriftNoise           = 1e-9;
            obj.LinearAccelerationNoise       = 1e-3;
            obj.MagneticDisturbanceNoise      = 1e-3;
            obj.ExpectedMagneticFieldStrength = 50;
        end
    end
end

classdef imufilter
    properties
        SampleRate
        State matrix              % [q_w q_x q_y q_z gx_b gy_b gz_b] (7×1)
        StateCovariance matrix
        AccelerometerNoise
        GyroscopeNoise
        GyroscopeDriftNoise
        LinearAccelerationNoise
    end
    methods
        function obj = imufilter()
            obj.SampleRate              = 100;
            obj.State                   = zeros(7, 1);
            obj.State(1)                = 1;
            obj.StateCovariance         = 1e-3 * eye(7);
            obj.AccelerometerNoise      = 1e-4;
            obj.GyroscopeNoise          = 1e-6;
            obj.GyroscopeDriftNoise     = 1e-9;
            obj.LinearAccelerationNoise = 1e-3;
        end
    end
end

classdef complementaryFilter
    properties
        SampleRate
        AccelerometerGain  % α  (0..1, smaller = trust gyro more)
        MagnetometerGain   % β
        Orientation matrix % 1×4 [w x y z]
    end
    methods
        function obj = complementaryFilter()
            obj.SampleRate          = 100;
            obj.AccelerometerGain   = 0.01;
            obj.MagnetometerGain    = 0.01;
            obj.Orientation         = zeros(1, 4);
            obj.Orientation(1)      = 1;
        end
    end
end

% insfilterMARG — EKF fusing IMU + GPS over a
% [quaternion (4); position (3); velocity (3); gyro-bias (3); accel-bias (3)]
% state vector (16 elements).
classdef insfilterMARG
    properties
        State matrix              % 16×1
        StateCovariance matrix    % 16×16
        IMUSampleRate
        GyroscopeNoise matrix
        AccelerometerNoise matrix
        GyroscopeBiasNoise matrix
        AccelerometerBiasNoise matrix
        GeomagneticVectorNED matrix  % 1×3 µT  reference field
    end
    methods
        function obj = insfilterMARG()
            obj.State                  = zeros(16, 1);
            obj.State(1)               = 1;          % q = identity
            obj.StateCovariance        = eye(16);
            obj.IMUSampleRate          = 100;
            obj.GyroscopeNoise         = 1e-6 * ones(1, 3);
            obj.AccelerometerNoise     = 1e-4 * ones(1, 3);
            obj.GyroscopeBiasNoise     = 1e-9 * ones(1, 3);
            obj.AccelerometerBiasNoise = 1e-9 * ones(1, 3);
            obj.GeomagneticVectorNED   = [27.555, -2.4169, -16.0849];
        end
    end
end
