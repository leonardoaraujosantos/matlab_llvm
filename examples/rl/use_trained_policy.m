% use_trained_policy.m — Reinforcement Learning Toolbox Tier-2.
%
% Demonstrates the policy-use accessors: after training a value-based agent,
% query its greedy policy on a specific observation with getAction /
% getMaxQValue, and extract a standalone greedy policy object with
% getGreedyPolicy (which then answers getAction the same way).
%
% These are the "deploy / use a trained agent" entry points: given an
% observation, what does the agent do, and how good does it think it is.

rng(0);

env = rlPredefinedEnv("CartPole-Discrete");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

agent = rlDQNAgent(obsInfo, actInfo);
agent.EpsilonDecay = 0.01;
agent.EpsilonMin   = 0.05;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 60;
trainOpts.MaxStepsPerEpisode = 200;
trainStats = train(agent, env, trainOpts);

% Query the trained agent on a near-upright pole leaning slightly right.
obs = [0.0; 0.0; 0.05; 0.0];
action = getAction(agent, obs);
qmax   = getMaxQValue(agent, obs);
fprintf('greedy action index: %.0f\n', action);
fprintf('max Q-value: %.3f\n', qmax);

% Extract a standalone greedy policy and confirm it agrees.
policy = getGreedyPolicy(agent);
policyAction = getAction(policy, obs);
fprintf('policy action index: %.0f\n', policyAction);
