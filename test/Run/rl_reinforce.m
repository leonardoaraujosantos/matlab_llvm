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
% The exact balanced-step count is a chaotic function of the learned weights:
% libm last-ULP differences in exp/log/sin/cos compound over 300 training
% episodes and through the cart-pole dynamics, so it diverges across platforms
% (391 on macOS, 352 on Linux).  Assert the platform-stable learning outcome
% instead — the trained greedy policy holds the pole far past the ~tens of steps
% an untrained policy manages, with a wide margin on either libm.
if balanced > 150
    fprintf('REINFORCE learned to balance the pole\n');
else
    fprintf('REINFORCE failed to balance the pole\n');
end
