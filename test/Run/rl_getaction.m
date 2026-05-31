% Reinforcement Learning Tier-2 gating test — policy-use accessors.
% Train a DQN agent, then query getAction / getMaxQValue and verify the
% greedy policy extracted by getGreedyPolicy agrees with the agent.
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
stats = train(agent, env, trainOpts);
obs = [0.0; 0.0; 0.05; 0.0];
action = getAction(agent, obs);
qmax   = getMaxQValue(agent, obs);
fprintf('action: %.0f\n', action);
fprintf('maxQ: %.3f\n', qmax);
policy = getGreedyPolicy(agent);
fprintf('policy action: %.0f\n', getAction(policy, obs));
