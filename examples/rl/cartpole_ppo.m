% cartpole_ppo.m — Reinforcement Learning Toolbox Tier-6 flagship.
%
% Mirrors the MathWorks "Train PPO Agent to Balance Cart-Pole" workflow. PPO
% (Proximal Policy Optimization) is the most widely-used on-policy RL algorithm.
% Unlike the off-policy DQN/DDPG/TD3 agents (which replay a buffer of past
% transitions), PPO learns from fresh on-policy rollouts:
%
%   1. Collect a rollout batch by running the current policy in the env.
%   2. Estimate each step's advantage with GAE(lambda) off a learned value
%      baseline V(s).
%   3. Run several epochs of a CLIPPED surrogate objective over the batch:
%      maximise min(r*A, clip(r, 1-eps, 1+eps)*A) where r = pi_new/pi_old. The
%      clip keeps each update close to the data-collecting policy, which is what
%      makes PPO stable.
%
% The softmax policy update and the value-function MSE both run on the SHIPPED
% Deep Learning Toolbox autodiff tape (dlarray/dlgradient); the RL runtime adds
% the rollout collection, the GAE advantage estimation and the clip.
%
% Cart-pole: observation = [x, xdot, theta, thetadot] (continuous, dim 4), two
% discrete actions (push left/right, +/-10 N), reward +1 per step the pole stays
% up, episode ends when |x|>2.4 m or |theta|>12 deg.

rng(0);

env = rlPredefinedEnv("CartPole-Discrete");

obsInfo = getObservationInfo(env);   % rlNumericSpec, dimension 4
actInfo = getActionInfo(env);        % rlFiniteSetSpec, 2 actions

% Default PPO agent — auto-builds a softmax-policy actor and a value-baseline
% critic from the observation/action specs.
agent = rlPPOAgent(obsInfo, actInfo);
agent.DiscountFactor = 0.99;
agent.LearnRate      = 0.001;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 80;   % PPO iterations (each collects a rollout batch)
trainOpts.MaxStepsPerEpisode = 500;

trainStats = train(agent, env, trainOpts);

% Greedy rollout of the trained policy (up to 500 steps).
balancedSteps = sim(agent, env);
fprintf('PPO greedy policy balanced the pole for %.0f steps\n', balancedSteps);
