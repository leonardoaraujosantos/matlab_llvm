% cartpole_reinforce.m — Reinforcement Learning Toolbox Tier-4.
%
% Mirrors the MathWorks REINFORCE policy-gradient (PG) workflow: a stochastic
% policy actor (softmax over discrete actions) trained directly to maximize
% expected return on the cart-pole environment.
%
% Per episode the agent rolls out by *sampling* actions from the policy,
% computes the discounted (variance-normalized) reward-to-go, and takes one
% policy-gradient step minimizing  −Σ logπ(aₜ|sₜ)·Ĝₜ.  That loss is assembled
% on the SHIPPED Deep Learning Toolbox autodiff tape (softmax → log → masked
% sum) and differentiated by dlgradient — the same reused tape as the DQN
% critic, with no autodiff numerics re-implemented.

rng(0);

env = rlPredefinedEnv("CartPole-Discrete");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Default PG agent — auto-builds a 24-unit softmax-policy actor MLP.
agent = rlPGAgent(obsInfo, actInfo);
agent.DiscountFactor = 0.99;
agent.LearnRate      = 0.01;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 300;
trainOpts.MaxStepsPerEpisode = 200;

trainStats = train(agent, env, trainOpts);

balancedSteps = sim(agent, env);
fprintf('REINFORCE greedy policy balanced the pole for %.0f steps\n', balancedSteps);
