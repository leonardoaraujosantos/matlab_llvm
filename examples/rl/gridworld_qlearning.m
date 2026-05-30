% gridworld_qlearning.m — Reinforcement Learning Toolbox Tier-1 headline.
%
% Mirrors the MathWorks "Train Q-Learning Agent to Solve Basic Grid World"
% example: build the predefined 5x5 BasicGridWorld MDP, create a table-backed
% Q critic, train a Q-learning agent, then a SARSA agent, and simulate the
% learned greedy policy.
%
% The whole tabular training loop (epsilon-greedy rollout + TD update) runs in
% the runtime over the environment's transition/reward tensors -- no neural
% network and no autodiff.  Tier-1 exposes the agent hyperparameters
% (DiscountFactor / LearnRate / Epsilon) directly on the agent object; in
% MathWorks MATLAB they are nested under agentOpts.EpsilonGreedyExploration
% and agentOpts.CriticOptimizerOptions.

rng(0);

% Predefined 5x5 grid world: start [2,1], terminal [5,5] (+10), a +5 jump
% from [2,4] to [4,4], obstacles block movement, every other step -1.
env = rlPredefinedEnv("BasicGridWorld");

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Table Q critic over the finite state/action sets.
qTable = rlTable(obsInfo, actInfo);
qFcn   = rlQValueFunction(qTable, obsInfo, actInfo);

% --- Q-learning agent (off-policy TD) ---
qAgent = rlQAgent(qFcn);
qAgent.DiscountFactor = 1;
qAgent.LearnRate      = 0.1;
qAgent.Epsilon        = 0.9;
qAgent.EpsilonDecay   = 0.01;
qAgent.EpsilonMin     = 0.01;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 200;
trainOpts.MaxStepsPerEpisode = 50;

qStats  = train(qAgent, env, trainOpts);
qReward = sim(qAgent, env);
fprintf('Q-learning greedy cumulative reward: %.1f\n', qReward);

% --- SARSA agent (on-policy TD) ---
sAgent = rlSARSAAgent(qFcn);
sAgent.DiscountFactor = 1;
sAgent.LearnRate      = 0.1;
sAgent.Epsilon        = 0.9;
sAgent.EpsilonDecay   = 0.01;
sAgent.EpsilonMin     = 0.01;

sStats  = train(sAgent, env, trainOpts);
sReward = sim(sAgent, env);
fprintf('SARSA greedy cumulative reward: %.1f\n', sReward);
