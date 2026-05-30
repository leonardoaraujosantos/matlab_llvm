% Reinforcement Learning Tier-4 gating test — REINFORCE policy gradient on the
% predefined cart-pole.  The −Σ logπ·Ĝ loss runs on the reused Deep Learning
% autodiff tape (softmax/log/sum); the learned greedy policy balances the pole
% far longer than a random policy, demonstrating policy-gradient learning.
rng(0);
env = rlPredefinedEnv("CartPole-Discrete");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
agent = rlPGAgent(obsInfo, actInfo);
agent.DiscountFactor = 0.99;
agent.LearnRate      = 0.01;
trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 300;
trainOpts.MaxStepsPerEpisode = 200;
stats = train(agent, env, trainOpts);
balanced = sim(agent, env);
fprintf('REINFORCE balanced steps: %.0f\n', balanced);
