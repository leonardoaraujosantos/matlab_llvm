% Reinforcement Learning Tier-1 gating test — tabular Q-learning + SARSA on
% the predefined 5x5 BasicGridWorld.  The runtime runs the whole epsilon-greedy
% TD training loop over the environment's transition/reward tensors; the learned
% greedy policy reaches the +10 terminal via the +5 jump for an optimal return
% of 11 (matching the MathWorks "Solve Basic Grid World" StopTrainingValue).
rng(0);

env = rlPredefinedEnv("BasicGridWorld");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

qTable = rlTable(obsInfo, actInfo);
qFcn   = rlQValueFunction(qTable, obsInfo, actInfo);

qAgent = rlQAgent(qFcn);
qAgent.DiscountFactor = 1;
qAgent.LearnRate      = 0.1;
qAgent.Epsilon        = 0.9;
qAgent.EpsilonDecay   = 0.01;
qAgent.EpsilonMin     = 0.01;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 200;
trainOpts.MaxStepsPerEpisode = 50;

qStats  = train(qAgent, env, trainOpts);
qReward = sim(qAgent, env);
fprintf('Q-learning reward: %.1f\n', qReward);

sAgent = rlSARSAAgent(qFcn);
sAgent.DiscountFactor = 1;
sAgent.LearnRate      = 0.1;
sAgent.Epsilon        = 0.9;
sAgent.EpsilonDecay   = 0.01;
sAgent.EpsilonMin     = 0.01;

sStats  = train(sAgent, env, trainOpts);
sReward = sim(sAgent, env);
fprintf('SARSA reward: %.1f\n', sReward);
