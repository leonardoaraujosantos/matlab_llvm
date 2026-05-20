% Model Predictive Control Toolbox — Tier-1 classdef umbrella.
% Auto-prepended by matlabc when the user input mentions `mpc(`,
% `mpcstate(`, or `mpcmove(`.  See tools/matlabc/main.cpp
% userMentionsExtClasses.  All heavy lifting lives in
% runtime/toolbox/mpc/runtime_mpc.cpp; the classdef bodies are thin
% wrappers around the `matlab_mpc_*` C-ABI entries.

classdef mpc
    properties
        % --- Plant (the constructor stamps A/B/C/Ts off the user-
        % supplied discrete-time `ss`).  Tier-1 requires a discrete
        % plant — call `c2d(ss_c, Ts)` first if you have continuous
        % dynamics.
        A
        B
        C
        Ts

        % --- Horizons (PredictionHorizon p, ControlHorizon m).
        p
        m

        % --- Standard four-term cost weights.
        Wy
        Wdu
        rho_eps

        % --- MV bounds — column vectors of length nu.
        umin
        umax

        % --- MV-rate bounds (Tier-6 §7.2).  Default empty → no rate
        % constraint.  Setting `dumin[k]` / `dumax[k]` adds 2·m rows
        % per MV channel `k` to the QP, constraining each Δu(j)[k]
        % directly.
        dumin
        dumax

        % --- MV-tracking weight + target (Tier-6 §7.3).  Cost gains
        % the `Σⱼ ‖Wu·(u(k+j) - u_target)‖²` term.  Default Wu = 0
        % (no MV tracking — the Tier-1/2 four-term cost).
        Wu
        u_target

        % --- Output (y) bounds — column vectors of length ny.
        % Tier-2 §3.4 / Tier-1 carve-down: 2·p·ny rows in the QP
        % inequality block enforce `ymin ≤ y(k+i) ≤ ymax` for
        % i = 1..p.  Defaults ±1e6 mean unconstrained.
        ymin
        ymax

        % --- Mixed input/output constraints  E·u + F·y + S·v ≤ G.
        % Tier-2 §3.1.  E (nE × nu), F (nE × ny), G (nE × 1).  All
        % empty by default; setconstraint() populates them.  At each
        % tick, mpc_tick translates these to z-space via Sx/Su/Su1.
        E
        F
        G

        % --- ECR (soft-constraint slack scaling).
        % Tier-2 §3.2.  Per-bound vectors V_*.  When V > 0 for a
        % given bound, that bound becomes soft: `bound ± ε·V` instead
        % of strict.  Default = 0 (hard).
        V_y_min
        V_y_max
        V_u_min
        V_u_max

        % --- Custom QP solver (Tier-4 §5.6).  When `UseCustomSolver`
        % is set and `CustomSolver` carries a MATLAB function handle
        % matching the signature `x = solver(H, f, A, b, x0)`, the
        % runtime calls it in place of the built-in KWIK.  Tier-4
        % ships the property storage; the dispatch hook (function-
        % handle ABI through the void* runtime call) is a follow-up
        % alongside multi-output handle support across `fmincon` /
        % `lsqnonlin` (Optim Tier-1 carve-down 2.4).
        CustomSolver
        UseCustomSolver

        % --- Binary / integer MVs (Tier-4 §5.7 Finite Control Set).
        % `mv_binary[k]` ≠ 0 marks MV k as restricted to {0, 1}; the
        % QP becomes MIQP and the runtime routes through branch-and-
        % bound over qp_kwik.  Tier-4 supports single-binary MVs;
        % multi-binary / general integer + cell-list value sets are a
        % follow-up.
        mv_binary

        % --- Output disturbance flag (Tier-2 §3.3).  When true, the
        % constructor augments the plant with an integrating output
        % disturbance per output (`d(k+1) = d(k)`), runs Kalman on
        % the augmented system, and cached A/B/C/Sx/Su/Su1/L all
        % carry the augmentation.  Default false (Tier-1 behavior).
        outdist
        nx_plant    % original (non-augmented) state dimension —
                    % needed by sim to drive the true plant.

        % --- Precomputed prediction matrices.
        Sx
        Su
        Su1
        H
        R
        L
    end

    methods
        function obj = mpc(plant, p, m)
            % `mpc(plant_d [, p [, m]])` — build the QP machinery for
            % a discrete-time state-space plant.  Defaults: p = 10,
            % m = 2 (matches MathWorks).  Tier-1 requires `plant` to
            % be a discrete-time `ss` (run `c2d` first if continuous).
            if nargin < 2, p = 10; end
            if nargin < 3, m = 2;  end
            % Type-hint assignments — Sema infers each property type
            % from these explicit MATLAB-side writes, then the C++
            % matlab_mpc_construct call below overwrites them with
            % the real values (discretized matrices, Cholesky factor,
            % Kalman gain, etc.).  Without the hints, the property
            % getter returns f64 instead of matrix and downstream
            % subscripting (e.g. `obj.Sx(1, 1)`) doesn't lower.
            obj.A   = plant.A;
            obj.B   = plant.B;
            obj.C   = plant.C;
            obj.Ts  = 0;
            obj.p   = p;
            obj.m   = m;
            obj.Wy      = zeros(1, 1);
            obj.Wdu     = zeros(1, 1);
            obj.rho_eps = 0;
            obj.umin = zeros(1, 1);
            obj.umax = zeros(1, 1);
            obj.dumin = zeros(1, 1);
            obj.dumax = zeros(1, 1);
            obj.Wu = zeros(1, 1);
            obj.u_target = zeros(1, 1);
            obj.ymin = zeros(1, 1);
            obj.ymax = zeros(1, 1);
            obj.E = zeros(1, 1);
            obj.F = zeros(1, 1);
            obj.G = zeros(1, 1);
            obj.V_y_min = zeros(1, 1);
            obj.V_y_max = zeros(1, 1);
            obj.V_u_min = zeros(1, 1);
            obj.V_u_max = zeros(1, 1);
            obj.outdist = 0;
            obj.nx_plant = 0;
            obj.CustomSolver = 0;
            obj.UseCustomSolver = 0;
            obj.mv_binary = zeros(1, 1);
            obj.Sx  = zeros(1, 1);
            obj.Su  = zeros(1, 1);
            obj.Su1 = zeros(1, 1);
            obj.H   = zeros(1, 1);
            obj.R   = zeros(1, 1);
            obj.L   = zeros(1, 1);
            matlab_mpc_construct(obj, plant, p, m);
        end

        % `mpcmove(obj, st, ym, r)` and `sim(obj, T, r)` are not
        % declared as classdef methods on purpose — they are
        % registered as builtins in Resolver.cpp and dispatched in
        % Lowering.cpp's class-pinned-first-arg path directly to the
        % `matlab_mpc_move` / `matlab_mpc_sim` runtime entries.
        % Routing via a classdef-method wrapper would land the inner
        % runtime call inside a function whose formal-parameter slots
        % carry `none` types, defeating the type-based dispatch.
    end
end

classdef mpcstate
    properties
        Plant      % nx × 1 plant state estimate xp
        LastMove   % nu × 1 last applied MV
        Dist       % ny × 1 output-disturbance estimate (Tier-2 §3.3)
    end
    methods
        function obj = mpcstate(nx, nu, ny)
            if nargin < 1, nx = 1; end
            if nargin < 2, nu = 1; end
            if nargin < 3, ny = 1; end
            obj.Plant    = zeros(nx, 1);
            obj.LastMove = zeros(nu, 1);
            obj.Dist     = zeros(ny, 1);
        end
    end
end

% generateExplicitMPC / mpcmoveExplicit — Tier-4 §5.1/5.2.  These
% are NOT declared as MATLAB functions on purpose — they are
% registered as builtins in Resolver.cpp and dispatched in
% Lowering.cpp's class-pinned-first-arg path directly to the
% `matlab_mpc_generate_explicit` / `matlab_mpc_move_explicit`
% runtime entries (same pattern as `mpcmove` / `sim` / `mpcmoveAdaptive`).

% nlmpc — Tier-5 Nonlinear MPC.  The plant is supplied as a MATLAB
% function handle `StateFcn` of signature `dxdt = stateFn(zxu)` where
% `zxu = [x; u]` is a packed vector (single-arg consistent with the
% Optim handle ABI).  At each tick `nlmpcmove(nlobj, st, ym, r)`
% builds a fmincon-style rollout cost over the decision variable
% (u trajectory of length m) and hands it to the shipped `fmincon`.
%
% Tier-5 simplifications (deferred to Tier-5b):
%   - OutputFcn defaults to identity (y = x[1:ny])
%   - CustomCostFcn defaults to tracking ‖r - y‖² + move-suppression
%   - Forward-Euler integration (RK4 is a follow-up)
%   - Analytic Jacobian intake deferred (FD gradients in fmincon)
%   - getCodeGenerationData / nlmpcMultistage deferred
classdef nlmpc
    properties
        nx             % plant state dimension
        nu             % MV dimension
        ny             % output dimension (Tier-5: y = x(1:ny))
        Ts             % sample time
        p              % prediction horizon
        m              % control horizon
        Wy             % ny × 1 output weights
        Wdu            % nu × 1 MV-rate weights
        rho_eps        % slack penalty
        umin           % nu × 1 MV lower bounds
        umax           % nu × 1 MV upper bounds
    end
    methods
        function obj = nlmpc(nx, ny, nu)
            if nargin < 1, nx = 1; end
            if nargin < 2, ny = 1; end
            if nargin < 3, nu = 1; end
            obj.nx = nx;
            obj.ny = ny;
            obj.nu = nu;
            obj.Ts = 0.1;
            obj.p  = 10;
            obj.m  = 2;
            obj.Wy      = ones(ny, 1);
            obj.Wdu     = 0.1 * ones(nu, 1);
            obj.rho_eps = 1e5;
            obj.umin = -1e6 * ones(nu, 1);
            obj.umax =  1e6 * ones(nu, 1);
        end
    end
end

% explicitMPC — Tier-4 §5.1/5.2 offline-tessellated MPC.  Built via
% `generateExplicitMPC(mpc, x_lo, x_hi, n_grid, r)` which solves the
% MPC QP at every grid point in the (x_lo, x_hi) hypercube and
% stores the resulting MV in a lookup table.  Run-time is pure
% O(grid_size · nx) nearest-neighbor lookup — no QP solver needed,
% suitable for embedded deployment.
%
% Tier-4 pragmatic form (grid tessellation).  The full Tøndel-
% Johansen-Bemporad multi-parametric QP that yields exact piecewise-
% affine regions is a research-grade follow-up.
classdef explicitMPC
    properties
        x_lo      % nx × 1 lower bound of the parameter hypercube
        x_hi      % nx × 1 upper bound
        n_grid    % scalar — points per dimension
        u_table   % (n_grid+1)^nx × nu  lookup table of optimal MVs
        nx
        nu
        ny
        Ts
        % Reference baked in at generation time.
        r_gen
    end
    methods
        function obj = explicitMPC()
            obj.x_lo    = zeros(1, 1);
            obj.x_hi    = zeros(1, 1);
            obj.n_grid  = 0;
            obj.u_table = zeros(1, 1);
            obj.nx      = 0;
            obj.nu      = 0;
            obj.ny      = 0;
            obj.Ts      = 0;
            obj.r_gen   = zeros(1, 1);
        end
    end
end

% mpcmoveopt — Tier-2 §3.7 run-time override carrier.  Pass as the
% 5th argument to `mpcmove(obj, st, ym, r, opt)` to override the
% cached MV / output bounds for this tick only.  Tier-2 ships with
% bound overrides only; weight overrides (`OutputWeights`, etc.) are
% a follow-up.
%
% Use_* flags signal "this field is set"; the runtime reads them via
% matlab_obj_get_f64 and falls back to the cached mpc properties
% when the flag is zero.
% mpcsimopt — Tier-6 §7.6 sim() override carrier.  Pass as the 4th
% argument to `sim(obj, T, r, opt)` to override per-sim behaviour.
% Tier-6 supports PlantInitialState only; OutputNoise / Disturbance
% / RefLookAhead / MDLookAhead are follow-ups.
classdef mpcsimopt
    properties
        PlantInitialState     % nx × 1 initial state override
        Use_PlantInitialState % flag scalar
    end
    methods
        function obj = mpcsimopt()
            obj.PlantInitialState = zeros(1, 1);
            obj.Use_PlantInitialState = 0;
        end
    end
end

classdef mpcmoveopt
    properties
        MVMin
        MVMax
        OutputMin
        OutputMax
        Use_MVMin
        Use_MVMax
        Use_OutputMin
        Use_OutputMax
    end
    methods
        function obj = mpcmoveopt()
            obj.MVMin     = zeros(1, 1);
            obj.MVMax     = zeros(1, 1);
            obj.OutputMin = zeros(1, 1);
            obj.OutputMax = zeros(1, 1);
            obj.Use_MVMin     = 0;
            obj.Use_MVMax     = 0;
            obj.Use_OutputMin = 0;
            obj.Use_OutputMax = 0;
        end
    end
end
