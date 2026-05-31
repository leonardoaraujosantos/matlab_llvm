% Reinforcement Learning Tier-6 gating test — SAC (Soft Actor-Critic) on the
% predefined pendulum swing-up.  SAC is the max-entropy continuous-control
% agent: a stochastic squashed-Gaussian actor (action = tanh(mean + std*eps) *
% limit, via the reparameterization trick) is trained to maximise
% Q(s,a) - alpha*log pi(a|s), with twin critics regressing a soft TD target
% that adds the entropy bonus.  The actor's log-prob (with the tanh-squash
% correction) is differentiated through the reused Deep Learning autodiff tape;
% the learned greedy policy swings the pole up and holds it near vertical.
rng(0);
env = rlPredefinedEnv("Pendulum-Continuous");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
agent = rlSACAgent(obsInfo, actInfo);
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
% ~-370, very tight across seeds, with a wide margin on either libm).
if r > -1000
    fprintf('SAC learned to swing up the pendulum\n');
else
    fprintf('SAC failed to swing up the pendulum\n');
end
