% pendulum_td3.m — Reinforcement Learning Toolbox Tier-6 flagship.
%
% Mirrors the MathWorks "Train TD3 Agent for Pendulum Swing-Up" workflow.
% TD3 (Twin Delayed Deep Deterministic policy gradient) is the modern,
% more-stable successor to DDPG. It keeps DDPG's deterministic actor and
% replay-based off-policy learning, but adds three fixes that prevent the
% critic's Q-value from blowing up:
%
%   1. Twin critics  — two Q networks are trained; the TD target uses the
%      MINIMUM of the two target critics, so an over-optimistic critic can't
%      mislead the policy.
%   2. Target-policy smoothing — clipped Gaussian noise is added to the target
%      action, so the critic can't latch onto a sharp, spurious Q peak.
%   3. Delayed updates — the actor and all target networks are updated once
%      every couple of critic steps, letting the critics settle first.
%
% Everything runs on the SHIPPED Deep Learning Toolbox autodiff tape
% (dlarray/dlgradient): the two critics' TD-error steps and the deterministic
% policy gradient (the actor's gradient flows through critic 1). The RL runtime
% only adds the Adam moment update, target soft-updates, and the replay/episode
% orchestration.
%
% Pendulum: observation = [cos(theta), sin(theta), thetadot] (continuous,
% dim 3), one continuous torque action in [-2, 2] N*m, reward
% -(theta^2 + 0.1*thetadot^2 + 0.001*u^2) per step. The pole starts hanging
% down; the learned policy pumps energy in to swing it up and holds it vertical.

rng(0);

env = rlPredefinedEnv("Pendulum-Continuous");

obsInfo = getObservationInfo(env);   % rlNumericSpec, dimension 3
actInfo = getActionInfo(env);        % rlNumericSpec, continuous torque +/-2

% Default TD3 agent — auto-builds a deterministic actor and twin Q(s,a) critics
% from the observation/action specs.
agent = rlTD3Agent(obsInfo, actInfo);
agent.DiscountFactor = 0.99;
agent.LearnRate      = 0.001;
agent.Tau            = 0.005;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 200;
trainOpts.MaxStepsPerEpisode = 200;

trainStats = train(agent, env, trainOpts);

% Greedy (noise-free) rollout of the trained policy from hanging-down.
totalReward = sim(agent, env);
fprintf('TD3 greedy swing-up return: %.0f\n', totalReward);
