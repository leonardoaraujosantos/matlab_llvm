% pendulum_ddpg.m — Reinforcement Learning Toolbox Tier-5 flagship.
%
% Mirrors the MathWorks "Train DDPG Agent to Swing Up and Balance Pendulum"
% workflow: build the predefined continuous-action pendulum environment,
% create a Deep Deterministic Policy Gradient agent (which auto-builds a
% deterministic actor and a Q(s,a) critic from the observation/action specs),
% train it with experience replay + target networks + Ornstein-Uhlenbeck
% exploration, then simulate the learned greedy policy.
%
% DDPG is the continuous-control keystone: unlike the discrete-action DQN /
% REINFORCE agents (which pick from a finite action set), the actor outputs a
% real-valued torque.  Both the critic's TD-error step and the deterministic
% policy gradient (the actor's gradient flows *through* the critic) run on the
% SHIPPED Deep Learning Toolbox autodiff tape (dlarray/dlgradient) — the RL
% runtime only adds the Adam optimizer-moment update, target soft-updates, and
% the replay/episode orchestration. No autodiff numerics are re-implemented.
%
% Pendulum: observation = [cos(theta), sin(theta), thetadot] (continuous,
% dim 3), one continuous torque action in [-2, 2] N*m, reward
% -(theta^2 + 0.1*thetadot^2 + 0.001*u^2) per step (0 is upright and still).
% The pole starts hanging down; the learned policy pumps energy in to swing it
% up and holds it near vertical.

rng(0);

env = rlPredefinedEnv("Pendulum-Continuous");

obsInfo = getObservationInfo(env);   % rlNumericSpec, dimension 3
actInfo = getActionInfo(env);        % rlNumericSpec, continuous torque +/-2

% Default DDPG agent — auto-builds a deterministic actor (obs -> relu -> tanh
% scaled to the torque limit) and a Q(s,a) critic ([obs;act] -> relu -> Q).
agent = rlDDPGAgent(obsInfo, actInfo);
agent.DiscountFactor = 0.99;
agent.LearnRate      = 0.001;
agent.Tau            = 0.005;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 200;
trainOpts.MaxStepsPerEpisode = 200;

trainStats = train(agent, env, trainOpts);

% Greedy (noise-free) rollout of the trained policy from hanging-down.
totalReward = sim(agent, env);
fprintf('DDPG greedy swing-up return: %.0f\n', totalReward);
