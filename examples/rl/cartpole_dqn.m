% cartpole_dqn.m — Reinforcement Learning Toolbox Tier-3 flagship.
%
% Mirrors the MathWorks "Train DQN Agent to Balance Cart-Pole" workflow:
% build the predefined continuous-state cart-pole environment, create a deep
% Q-network agent (which auto-builds an MLP critic from the observation/action
% specs), train it with experience replay + a target network + epsilon-greedy
% exploration, then simulate the learned greedy policy.
%
% This is the keystone deep-RL slice: the critic's forward pass and the
% TD-error gradient step run on the SHIPPED Deep Learning Toolbox autodiff
% tape (dlarray/dlgradient) — the RL runtime builds the network on that tape
% and only adds the Adam optimizer-moment update and the replay/episode
% orchestration. No autodiff numerics are re-implemented.
%
% Cart-pole: observation = [x, xdot, theta, thetadot] (continuous, dim 4),
% two discrete actions (push left / right, +/-10 N), reward +1 per step the
% pole stays up, episode ends when |x|>2.4 m or |theta|>12 deg.

rng(0);

env = rlPredefinedEnv("CartPole-Discrete");

obsInfo = getObservationInfo(env);   % rlNumericSpec, dimension 4
actInfo = getActionInfo(env);        % rlFiniteSetSpec, 2 actions

% Default DQN agent — auto-builds a 24-unit MLP critic (obs -> relu -> Q).
agent = rlDQNAgent(obsInfo, actInfo);
agent.DiscountFactor = 0.99;
agent.LearnRate      = 0.001;
agent.Epsilon        = 1.0;
agent.EpsilonDecay   = 0.01;
agent.EpsilonMin     = 0.05;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 150;
trainOpts.MaxStepsPerEpisode = 200;

trainStats = train(agent, env, trainOpts);

% Greedy rollout of the trained policy (up to 500 steps).
balancedSteps = sim(agent, env);
fprintf('DQN greedy policy balanced the pole for %.0f steps\n', balancedSteps);
