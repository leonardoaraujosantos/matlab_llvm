% Reinforcement Learning Tier-5 gating test — DDPG continuous control on the
% predefined pendulum swing-up.  The critic TD-error step and the deterministic
% policy gradient (actor gradient flows through the critic) both run on the
% reused Deep Learning autodiff tape; the learned greedy policy swings the pole
% up and holds it near vertical, scoring far above an untrained policy.
rng(0);
env = rlPredefinedEnv("Pendulum-Continuous");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
agent = rlDDPGAgent(obsInfo, actInfo);
agent.DiscountFactor = 0.99;
agent.LearnRate      = 0.001;
agent.Tau            = 0.005;
trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 200;
trainOpts.MaxStepsPerEpisode = 200;
stats = train(agent, env, trainOpts);
r = sim(agent, env);
% The exact return is a chaotic, libm-dependent value (the swing-up trajectory
% and learned weights diverge across platforms), so assert the platform-stable
% learning outcome instead: the trained actor swings the pole up and holds it,
% scoring far above the ~-1600 an untrained policy manages (trained ≈ -380,
% with a wide margin on either libm).
if r > -900
    fprintf('DDPG learned to swing up the pendulum\n');
else
    fprintf('DDPG failed to swing up the pendulum\n');
end
