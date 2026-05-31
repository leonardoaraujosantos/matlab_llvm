% pendulum_sac.m — Reinforcement Learning Toolbox Tier-6 flagship.
%
% Mirrors the MathWorks "Train SAC Agent for Pendulum Swing-Up" workflow.
% SAC (Soft Actor-Critic) is the state-of-the-art off-policy continuous-control
% algorithm. Unlike DDPG/TD3's deterministic actor, SAC learns a STOCHASTIC
% squashed-Gaussian policy and maximises a MAX-ENTROPY objective — expected
% return plus the policy's entropy — which makes exploration intrinsic and
% training markedly more stable:
%
%   - Actor: shared trunk -> mean and log-std heads; the action is sampled as
%     a = tanh(mean + std * eps) * limit (the reparameterization trick), so the
%     sample is differentiable w.r.t. the policy parameters.
%   - Twin critics regress the soft TD target
%       y = r + gamma * ( min(Qt1, Qt2) - alpha * log pi(a'|s') ),  a' ~ pi.
%   - The actor maximises Q1(s,a) - alpha * log pi(a|s).
%
% The actor's log-prob (with the tanh-squash change-of-variables correction)
% and both critics' losses run on the SHIPPED Deep Learning Toolbox autodiff
% tape (dlarray/dlgradient). The entropy temperature alpha is held fixed (the
% canonical fixed-coefficient SAC variant).
%
% Pendulum: observation = [cos(theta), sin(theta), thetadot] (continuous,
% dim 3), one continuous torque action in [-2, 2] N*m, reward
% -(theta^2 + 0.1*thetadot^2 + 0.001*u^2) per step. The pole starts hanging
% down; the learned policy swings it up and holds it vertical.

rng(0);

env = rlPredefinedEnv("Pendulum-Continuous");

obsInfo = getObservationInfo(env);   % rlNumericSpec, dimension 3
actInfo = getActionInfo(env);        % rlNumericSpec, continuous torque +/-2

% Default SAC agent — auto-builds a squashed-Gaussian actor and twin Q(s,a)
% critics from the observation/action specs.
agent = rlSACAgent(obsInfo, actInfo);
agent.DiscountFactor = 0.99;
agent.LearnRate      = 0.001;
agent.Tau            = 0.005;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 200;
trainOpts.MaxStepsPerEpisode = 200;

trainStats = train(agent, env, trainOpts);

% Greedy (mean-action, noise-free) rollout of the trained policy.
totalReward = sim(agent, env);
fprintf('SAC greedy swing-up return: %.0f\n', totalReward);
