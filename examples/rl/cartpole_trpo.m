% cartpole_trpo.m — Reinforcement Learning Toolbox: TRPO flagship.
%
% Mirrors the MathWorks "Train TRPO Agent" workflow. TRPO (Trust Region Policy
% Optimization) is the principled predecessor of PPO: rather than PPO's
% first-order clipped surrogate, it takes a second-order NATURAL-gradient step
% inside a hard KL trust region, which guarantees monotonic-ish improvement.
%
%   1. Collect on-policy rollouts; estimate advantages with GAE(lambda) off a
%      learned value baseline.
%   2. Policy gradient g = E[A * grad log pi].
%   3. Natural-gradient direction x = F^{-1} g, where F is the Fisher
%      information matrix, solved by CONJUGATE GRADIENT using Fisher-vector
%      products. (FVP(v) is obtained from the reverse-mode KL gradient:
%      FVP(v) ~= grad_theta KL(pi_old || pi_{theta+eps*v}) / eps, since the KL
%      gradient is zero at theta_old and its Hessian is the Fisher matrix.)
%   4. Scale x to the trust-region boundary: alpha = sqrt(2*delta / x'Fx).
%   5. BACKTRACKING line search: shrink the step until the KL constraint
%      (<= delta) holds and the surrogate objective improves.
%
% The policy gradient, the KL gradient (for the Fisher-vector products), and
% the value-baseline MSE all run on the SHIPPED Deep Learning autodiff tape.
%
% Cart-pole: observation = [x, xdot, theta, thetadot] (continuous, dim 4), two
% discrete actions (push left/right), reward +1 per step the pole stays up,
% episode ends when |x|>2.4 m or |theta|>12 deg.

rng(0);

env = rlPredefinedEnv("CartPole-Discrete");

obsInfo = getObservationInfo(env);   % rlNumericSpec, dimension 4
actInfo = getActionInfo(env);        % rlFiniteSetSpec, 2 actions

% Default TRPO agent — a discrete softmax policy + a value-baseline critic.
agent = rlTRPOAgent(obsInfo, actInfo);
agent.LearnRate = 0.001;   % value-baseline learn rate
agent.KLLimit   = 0.01;    % trust-region size (max KL per update)

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 120;   % TRPO iterations (each collects a rollout batch)
trainOpts.MaxStepsPerEpisode = 500;

trainStats = train(agent, env, trainOpts);

% Greedy rollout of the trained policy (up to 500 steps).
balancedSteps = sim(agent, env);
fprintf('TRPO greedy policy balanced the pole for %.0f steps\n', balancedSteps);
