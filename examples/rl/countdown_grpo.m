% countdown_grpo.m — Reinforcement Learning Toolbox: GRPO flagship.
%
% GRPO (Group Relative Policy Optimization) is the algorithm DeepSeek
% introduced to post-train reasoning LLMs. Its defining move is to ELIMINATE
% the critic (value) network that PPO/SAC require: instead of a second large
% model estimating expected reward, GRPO samples a GROUP of candidate answers
% to the same prompt, scores them with a rule-based verifier, and uses the
% group's own statistics as the baseline.
%
% This example exercises GRPO on a small but faithful analogue of an LLM
% reasoning task — a Countdown-style arithmetic puzzle that has exactly the
% properties GRPO is built for: discrete actions, cheap parallel resets, and a
% sparse OUTCOME reward checked by a verifier (right answer or not). Starting
% from 0, the agent applies two operations (pick a digit 1-5 and one of +,-,*)
% to try to hit a target. For each target the agent samples a group of M=24
% completions; the reward is 1 iff the final value equals the target.
%
%   - Group-relative advantage:  A_i = (r_i - mean(r)) / std(r)   (no critic!)
%   - Clipped surrogate policy update (as in PPO) + an explicit
%     KL-to-reference penalty, both on the SHIPPED Deep Learning autodiff tape.
%
% The learned greedy policy solves most of the 8-target set; an untrained
% policy solves none.

rng(0);

env = rlPredefinedEnv("Countdown-Discrete");

obsInfo = getObservationInfo(env);   % [acc, target, step], dimension 3
actInfo = getActionInfo(env);        % 15 discrete (digit, op) actions

% Default GRPO agent — a discrete softmax policy; no critic network is built.
agent = rlGRPOAgent(obsInfo, actInfo);
agent.LearnRate = 0.003;
agent.GroupSize = 24;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 800;   % GRPO iterations (each samples a group per target)
trainOpts.MaxStepsPerEpisode = 2;

trainStats = train(agent, env, trainOpts);

% Greedy evaluation: how many of the 8 target puzzles the policy solves.
solved = sim(agent, env);
fprintf('GRPO greedy policy solved %.0f / 8 Countdown puzzles\n', solved);
