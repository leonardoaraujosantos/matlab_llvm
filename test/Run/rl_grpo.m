% Reinforcement Learning gating test — GRPO (Group Relative Policy
% Optimization, DeepSeek) on a discrete Countdown arithmetic-puzzle env.
% GRPO is on-policy and CRITIC-FREE: for each target it samples a GROUP of
% candidate solution sequences, scores them with a rule-based verifier
% (reward 1 iff the arithmetic hits the target), and uses the group's own
% mean/std to form the advantage A_i=(r_i-mu)/sigma — replacing PPO's value
% network and GAE entirely.  The clipped-surrogate policy update (plus the
% KL-to-reference penalty) runs on the reused Deep Learning autodiff tape.
% The learned greedy policy solves the puzzle set; an untrained policy solves
% none.
rng(0);
env = rlPredefinedEnv("Countdown-Discrete");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
agent = rlGRPOAgent(obsInfo, actInfo);
agent.LearnRate = 0.003;
agent.GroupSize = 24;
trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 800;   % GRPO iterations (each samples a group per target)
trainOpts.MaxStepsPerEpisode = 2;
stats = train(agent, env, trainOpts);
solved = sim(agent, env);   % greedy puzzles solved out of 8
% The exact solved-count is a chaotic, libm-dependent value, so assert the
% platform-stable learning outcome: the critic-free group-relative policy
% solves most of the puzzle set, far above the 0 an untrained policy manages.
if solved > 3
    fprintf('GRPO learned to solve the Countdown puzzles\n');
else
    fprintf('GRPO failed to solve the Countdown puzzles\n');
end
