% Reinforcement Learning Toolbox — classdef umbrella (Tier 1).
% Auto-prepended by matlabc when the user input mentions any RL symbol
% (`rlPredefinedEnv`, `rlMDPEnv`, `rlNumericSpec`, `rlFiniteSetSpec`,
% `rlTable`, `rlQValueFunction`, `rlQAgent`, `rlSARSAAgent`, ...).
%
% Tier 1 = the tabular, autodiff-free slice: grid-world / MDP environments,
% a table-backed Q critic, and the Q-learning + SARSA agents whose updates
% are closed-form TD steps.  The whole training loop runs inside the runtime
% (`matlab_rl_train`) over the environment's transition/reward tensors — no
% MATLAB-side episode loop and no neural network.  See
% docs/reinforcement_learning_toolbox_roadmap.md.
%
% Every classdef ships a *neutral* zero-arg constructor only; the user-facing
% forms (`rlTable(obsInfo,actInfo)`, `rlQAgent(critic)`, ...) and the free
% functions (`rlPredefinedEnv`, `getObservationInfo`, `train`, `sim`) are
% intercepted in Lowering.cpp and populated by `matlab_rl_*` runtime calls.

% ===== Observation / action specs =========================================
classdef rlFiniteSetSpec
    properties
        Elements matrix          % 1×N or N×1 set of admissible values
        Dimension matrix         % 1×2
    end
    methods
        function obj = rlFiniteSetSpec()
            obj.Elements  = [1];
            obj.Dimension = [1, 1];
        end
    end
end

classdef rlNumericSpec
    properties
        Dimension matrix         % 1×2 signal size
        LowerLimit matrix
        UpperLimit matrix
    end
    methods
        function obj = rlNumericSpec()
            obj.Dimension  = [1, 1];
            obj.LowerLimit = [-1e308];
            obj.UpperLimit = [1e308];
        end
    end
end

% ===== Environment ========================================================
% rlMDPEnv carries the finite MDP: NumStates, NumActions, a flattened
% transition tensor T (NumStates*NumActions × NumStates, row = (s-1)*A+a) and
% a reward tensor R laid out the same way, plus the start/terminal states.
% rlPredefinedEnv("BasicGridWorld") fills these in the runtime.
classdef rlMDPEnv
    properties
        T matrix                 % (S*A)×S transition probabilities
        R matrix                 % (S*A)×S immediate rewards
        NumStates
        NumActions
        StartState
        TerminalState
        GridRows
        GridCols
    end
    methods
        function obj = rlMDPEnv()
            obj.T             = zeros(1, 1);
            obj.R             = zeros(1, 1);
            obj.NumStates     = 1;
            obj.NumActions    = 1;
            obj.StartState    = 1;
            obj.TerminalState = 1;
            obj.GridRows      = 0;
            obj.GridCols      = 0;
        end
    end
end

% rlFunctionEnv wraps user step/reset handles (Tier-1 single-step surface;
% the full training loop over handles is a documented carve to later tiers).
classdef rlFunctionEnv
    properties
        ObservationDimension matrix
        ActionElements matrix
        State matrix             % current internal state
    end
    methods
        function obj = rlFunctionEnv()
            obj.ObservationDimension = [1, 1];
            obj.ActionElements       = [0];
            obj.State                = zeros(1, 1);
        end
    end
end

% ===== Approximators ======================================================
classdef rlTable
    properties
        Table matrix             % S×A action-value table
    end
    methods
        function obj = rlTable()
            obj.Table = zeros(1, 1);
        end
    end
end

classdef rlQValueFunction
    properties
        QTable matrix            % S×A copy of the table approximation
        NumStates
        NumActions
    end
    methods
        function obj = rlQValueFunction()
            obj.QTable     = zeros(1, 1);
            obj.NumStates  = 1;
            obj.NumActions = 1;
        end
    end
end

% ===== Options ============================================================
classdef rlOptimizerOptions
    properties
        LearnRate
        L2RegularizationFactor
    end
    methods
        function obj = rlOptimizerOptions()
            obj.LearnRate              = 0.01;
            obj.L2RegularizationFactor = 0;
        end
    end
end

% Agent options carry the discount, the epsilon-greedy exploration struct,
% and the critic optimizer options.  EpsilonGreedyExploration is a plain
% struct so the documented `opts.EpsilonGreedyExploration.Epsilon = x`
% two-level assignment lowers through the supported struct-field-store path.
classdef rlQAgentOptions
    properties
        DiscountFactor
        EpsilonGreedyExploration
        CriticOptimizerOptions
        LearnRate
    end
    methods
        function obj = rlQAgentOptions()
            obj.DiscountFactor = 0.99;
            obj.EpsilonGreedyExploration = struct( ...
                'Epsilon', 0.9, 'EpsilonDecay', 0.005, 'EpsilonMin', 0.01);
            obj.CriticOptimizerOptions = rlOptimizerOptions();
            obj.LearnRate = 0.01;
        end
    end
end

classdef rlSARSAAgentOptions
    properties
        DiscountFactor
        EpsilonGreedyExploration
        CriticOptimizerOptions
        LearnRate
    end
    methods
        function obj = rlSARSAAgentOptions()
            obj.DiscountFactor = 0.99;
            obj.EpsilonGreedyExploration = struct( ...
                'Epsilon', 0.9, 'EpsilonDecay', 0.005, 'EpsilonMin', 0.01);
            obj.CriticOptimizerOptions = rlOptimizerOptions();
            obj.LearnRate = 0.01;
        end
    end
end

% ===== Agents =============================================================
% Agents carry a copy of the critic Q-table plus the resolved scalar
% hyperparameters (discount / learn-rate / epsilon schedule).  `train`
% updates QTable in place via the runtime; `getCritic`/`getLearnableParameters`
% read it back.  IsSARSA selects the on-policy update inside the runtime.
classdef rlQAgent
    properties
        QTable matrix
        NumStates
        NumActions
        DiscountFactor
        LearnRate
        Epsilon
        EpsilonDecay
        EpsilonMin
        IsSARSA
    end
    methods
        function obj = rlQAgent()
            obj.QTable         = zeros(1, 1);
            obj.NumStates      = 1;
            obj.NumActions     = 1;
            obj.DiscountFactor = 0.99;
            obj.LearnRate      = 0.01;
            obj.Epsilon        = 0.9;
            obj.EpsilonDecay   = 0.005;
            obj.EpsilonMin     = 0.01;
            obj.IsSARSA        = 0;
        end
    end
end

classdef rlSARSAAgent
    properties
        QTable matrix
        NumStates
        NumActions
        DiscountFactor
        LearnRate
        Epsilon
        EpsilonDecay
        EpsilonMin
        IsSARSA
    end
    methods
        function obj = rlSARSAAgent()
            obj.QTable         = zeros(1, 1);
            obj.NumStates      = 1;
            obj.NumActions     = 1;
            obj.DiscountFactor = 0.99;
            obj.LearnRate      = 0.01;
            obj.Epsilon        = 0.9;
            obj.EpsilonDecay   = 0.005;
            obj.EpsilonMin     = 0.01;
            obj.IsSARSA        = 1;
        end
    end
end

% ===== Deep value-based agent (Tier 3) ====================================
% rlDQNAgent(obsInfo, actInfo) auto-builds a 2-layer MLP critic (obsDim → H
% relu → numActions); the network weights / Adam moments / target copies are
% added as matrix fields by the runtime initializer.  Only the scalar
% hyperparameters are declared here.  The TD-error gradient step reuses the
% Deep Learning Toolbox autodiff tape.
classdef rlDQNAgent
    properties
        ObsDim
        NumActions
        HiddenSize
        DiscountFactor
        LearnRate
        Epsilon
        EpsilonDecay
        EpsilonMin
    end
    methods
        function obj = rlDQNAgent()
            obj.ObsDim         = 1;
            obj.NumActions     = 1;
            obj.HiddenSize     = 24;
            obj.DiscountFactor = 0.99;
            obj.LearnRate      = 0.001;
            obj.Epsilon        = 1.0;
            obj.EpsilonDecay   = 0.005;
            obj.EpsilonMin     = 0.01;
        end
    end
end

% ===== Policy-gradient agent (Tier 4) ====================================
% rlPGAgent(obsInfo, actInfo) auto-builds a softmax-policy actor MLP
% (obsDim → H relu → numActions logits).  REINFORCE update over the reused
% autodiff tape; no epsilon (exploration is via action sampling).
classdef rlPGAgent
    properties
        ObsDim
        NumActions
        HiddenSize
        DiscountFactor
        LearnRate
        Epsilon
        EpsilonDecay
        EpsilonMin
    end
    methods
        function obj = rlPGAgent()
            obj.ObsDim         = 1;
            obj.NumActions     = 1;
            obj.HiddenSize     = 24;
            obj.DiscountFactor = 0.99;
            obj.LearnRate      = 0.01;
            obj.Epsilon        = 0;
            obj.EpsilonDecay   = 0;
            obj.EpsilonMin     = 0;
        end
    end
end

% ===== Continuous-control agent (Tier 5) =================================
% rlDDPGAgent(obsInfo, actInfo) auto-builds a deterministic actor + a Q(s,a)
% critic (each with a soft-updated target copy); the network weights / Adam
% moments / target copies are added as matrix fields by the runtime.  Critic
% TD-MSE + deterministic-policy-gradient actor step both run on the reused
% Deep Learning autodiff tape.
classdef rlDDPGAgent
    properties
        ObsDim
        ActDim
        ActLimit
        HiddenSize
        DiscountFactor
        LearnRate
        Tau
    end
    methods
        function obj = rlDDPGAgent()
            obj.ObsDim         = 1;
            obj.ActDim         = 1;
            obj.ActLimit       = 1;
            obj.HiddenSize     = 32;
            obj.DiscountFactor = 0.99;
            obj.LearnRate      = 0.001;
            obj.Tau            = 0.005;
        end
    end
end

% rlTD3Agent — Twin Delayed DDPG.  Same continuous-control interface as DDPG
% (deterministic actor + Q critic over the pendulum env); the runtime adds a
% second critic, target-policy smoothing and delayed actor/target updates.
classdef rlTD3Agent
    properties
        ObsDim
        ActDim
        ActLimit
        HiddenSize
        DiscountFactor
        LearnRate
        Tau
    end
    methods
        function obj = rlTD3Agent()
            obj.ObsDim         = 1;
            obj.ActDim         = 1;
            obj.ActLimit       = 1;
            obj.HiddenSize     = 32;
            obj.DiscountFactor = 0.99;
            obj.LearnRate      = 0.001;
            obj.Tau            = 0.005;
        end
    end
end

% rlPPOAgent — Proximal Policy Optimization (on-policy actor-critic).  Discrete
% cart-pole: the runtime collects a rollout batch, estimates GAE advantages off
% a learned value baseline, and runs several clipped-surrogate epochs per batch.
classdef rlPPOAgent
    properties
        ObsDim
        NumActions
        DiscountFactor
        LearnRate
    end
    methods
        function obj = rlPPOAgent()
            obj.ObsDim         = 1;
            obj.NumActions     = 2;
            obj.DiscountFactor = 0.99;
            obj.LearnRate      = 0.001;
        end
    end
end

% rlSACAgent — Soft Actor-Critic (max-entropy continuous control).  Stochastic
% squashed-Gaussian actor + twin Q critics over the pendulum env; trains with a
% fixed entropy coefficient on the reused autodiff tape.
classdef rlSACAgent
    properties
        ObsDim
        ActDim
        ActLimit
        HiddenSize
        DiscountFactor
        LearnRate
        Tau
    end
    methods
        function obj = rlSACAgent()
            obj.ObsDim         = 1;
            obj.ActDim         = 1;
            obj.ActLimit       = 1;
            obj.HiddenSize     = 32;
            obj.DiscountFactor = 0.99;
            obj.LearnRate      = 0.001;
            obj.Tau            = 0.005;
        end
    end
end

% ===== Policy object (Tier 2) =============================================
% rlMaxQPolicy — the greedy policy extracted from a value-based agent via
% getGreedyPolicy.  Carries a copy of the agent's network (or Q table), added
% as matrix fields by the runtime; getAction dispatches the same as on an agent.
classdef rlMaxQPolicy
    properties
        Kind
    end
    methods
        function obj = rlMaxQPolicy()
            obj.Kind = 0;
        end
    end
end

classdef rlTrainingOptions
    properties
        MaxStepsPerEpisode
        MaxEpisodes
        ScoreAveragingWindowLength
        StopTrainingValue
    end
    methods
        function obj = rlTrainingOptions()
            obj.MaxStepsPerEpisode         = 500;
            obj.MaxEpisodes                = 500;
            obj.ScoreAveragingWindowLength = 5;
            obj.StopTrainingValue          = 1e9;
        end
    end
end

classdef rlSimulationOptions
    properties
        MaxSteps
        NumSimulations
    end
    methods
        function obj = rlSimulationOptions()
            obj.MaxSteps       = 500;
            obj.NumSimulations = 1;
        end
    end
end
