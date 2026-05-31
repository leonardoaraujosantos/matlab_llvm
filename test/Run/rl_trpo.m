% Reinforcement Learning gating test — TRPO (Trust Region Policy Optimization)
% on the discrete cart-pole.  TRPO is the on-policy natural-gradient method:
% instead of PPO's clipped Adam step it takes a NATURAL-gradient step
% x = F^{-1} g (g = policy gradient, F = Fisher matrix) solved by conjugate
% gradient, scales it to the KL trust-region boundary, then backtracks until
% the KL constraint holds and the surrogate improves.  The Fisher-vector
% products use the reverse-mode KL gradient on the reused autodiff tape; a
% value baseline (GAE) is fit by MSE.  The learned greedy policy balances the
% pole far longer than an untrained policy (~10-20 steps).
rng(0);
env = rlPredefinedEnv("CartPole-Discrete");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
agent = rlTRPOAgent(obsInfo, actInfo);
agent.LearnRate = 0.001;   % value-baseline learn rate
agent.KLLimit   = 0.01;    % trust-region size
trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 120;   % TRPO iterations (each collects a rollout batch)
trainOpts.MaxStepsPerEpisode = 500;
stats = train(agent, env, trainOpts);
balanced = sim(agent, env);
% The exact balanced-step count is a chaotic, libm-dependent value, so assert
% the platform-stable learning outcome: the trust-region policy balances the
% pole many times longer than the ~10-20 steps of an untrained policy.
if balanced > 50
    fprintf('TRPO learned to balance the pole\n');
else
    fprintf('TRPO failed to balance the pole\n');
end
