% Reinforcement Learning Tier-6 gating test — PPO (Proximal Policy
% Optimization) on the predefined discrete cart-pole.  PPO is on-policy: each
% iteration collects a fresh rollout batch, estimates advantages with GAE(lambda)
% off a learned value baseline, and runs several clipped-surrogate epochs over
% the batch.  The policy + value updates run on the reused Deep Learning
% autodiff tape; the learned greedy policy balances the pole far longer than an
% untrained policy (~10-20 steps).
rng(0);
env = rlPredefinedEnv("CartPole-Discrete");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
agent = rlPPOAgent(obsInfo, actInfo);
agent.DiscountFactor = 0.99;
agent.LearnRate      = 0.001;
trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 80;   % PPO iterations (each collects a rollout batch)
trainOpts.MaxStepsPerEpisode = 500;
stats = train(agent, env, trainOpts);
balanced = sim(agent, env);
% The exact balanced-step count is a chaotic, libm-dependent value, so assert
% the platform-stable learning outcome: the trained greedy policy balances the
% pole many times longer than the ~10-20 steps of an untrained policy.  The
% threshold sits well above random yet far below the ~138-500 the trained
% policy reaches, giving margin on either libm.
if balanced > 50
    fprintf('PPO learned to balance the pole\n');
else
    fprintf('PPO failed to balance the pole\n');
end
