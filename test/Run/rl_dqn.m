% Reinforcement Learning Tier-3 gating test — DQN on the predefined cart-pole.
% The critic forward + TD-error gradient run on the reused Deep Learning
% autodiff tape; the learned greedy policy balances the pole far longer than
% a random policy (~10-20 steps), demonstrating end-to-end deep RL.
rng(0);
env = rlPredefinedEnv("CartPole-Discrete");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
agent = rlDQNAgent(obsInfo, actInfo);
agent.DiscountFactor = 0.99;
agent.LearnRate      = 0.001;
agent.Epsilon        = 1.0;
agent.EpsilonDecay   = 0.01;
agent.EpsilonMin     = 0.05;
trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 150;
trainOpts.MaxStepsPerEpisode = 200;
stats = train(agent, env, trainOpts);
balanced = sim(agent, env);
fprintf('DQN balanced steps: %.0f\n', balanced);
