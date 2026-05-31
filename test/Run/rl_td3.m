% Reinforcement Learning Tier-6 gating test — TD3 (Twin Delayed DDPG) on the
% predefined pendulum swing-up.  TD3 layers three fixes on DDPG that tame
% Q-value overestimation: twin critics (the TD target takes the minimum of two
% target critics), target-policy smoothing (clipped noise on the target
% action), and delayed actor/target updates.  All of it runs on the reused
% Deep Learning autodiff tape; the learned greedy policy swings the pole up and
% holds it near vertical, scoring far above an untrained policy.
rng(0);
env = rlPredefinedEnv("Pendulum-Continuous");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
agent = rlTD3Agent(obsInfo, actInfo);
agent.DiscountFactor = 0.99;
agent.LearnRate      = 0.001;
agent.Tau            = 0.005;
trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 200;
trainOpts.MaxStepsPerEpisode = 200;
stats = train(agent, env, trainOpts);
r = sim(agent, env);
% The exact return is a chaotic, libm-dependent value, so assert the
% platform-stable learning outcome: the trained actor swings the pole up and
% holds it, scoring far above the ~-1600 an untrained policy manages (trained
% ~-380, with a wide margin on either libm).
if r > -1000
    fprintf('TD3 learned to swing up the pendulum\n');
else
    fprintf('TD3 failed to swing up the pendulum\n');
end
