% mdp_qlearning.m — Reinforcement Learning Toolbox Tier-1.
%
% Mirrors the MathWorks "Train Reinforcement Learning Agent in MDP
% Environment" example: an 8-state, 2-action ("up"/"down") deterministic
% Markov Decision Process, solved with a tabular Q-learning agent.
%
% The MathWorks form chains  mdp = createMDP(8,["up";"down"]);  then fills the
% 3-D transition/reward tensors with  mdp.T(s,s',a)=...  /  mdp.R(s,s',a)=...
% and wraps it as  env = rlMDPEnv(mdp).  3-D indexed-property assignment is a
% documented frontend carve, so here we build the (deterministic) MDP from an
% S-by-A next-state table and a matching reward table and pass them directly to
% rlMDPEnv.  States 7 and 8 are terminal (their rows self-loop).
%
% MDP graph (action 1 = up, action 2 = down):
%   1: up->2 (r3)  down->3 (r1)        5: up->7 (r1)  down->8 (r9)
%   2: up->4 (r2)  down->5 (r1)        6: up->7 (r5)  down->8 (r1)
%   3: up->5 (r2)  down->6 (r4)        7,8: terminal
%   4: up->7 (r3)  down->8 (r2)
% Optimal return from state 1 is 1->2->5->8 = 3 + 1 + 9 = 13.

rng(0);

% Next-state table (1-based) and reward table, S=8 rows, A=2 cols [up down].
NS = zeros(8, 2);
RW = zeros(8, 2);
NS(1,1) = 2; NS(1,2) = 3; RW(1,1) = 3; RW(1,2) = 1;
NS(2,1) = 4; NS(2,2) = 5; RW(2,1) = 2; RW(2,2) = 1;
NS(3,1) = 5; NS(3,2) = 6; RW(3,1) = 2; RW(3,2) = 4;
NS(4,1) = 7; NS(4,2) = 8; RW(4,1) = 3; RW(4,2) = 2;
NS(5,1) = 7; NS(5,2) = 8; RW(5,1) = 1; RW(5,2) = 9;
NS(6,1) = 7; NS(6,2) = 8; RW(6,1) = 5; RW(6,2) = 1;
NS(7,1) = 7; NS(7,2) = 7;          % terminal self-loop
NS(8,1) = 8; NS(8,2) = 8;          % terminal self-loop

env = rlMDPEnv(NS, RW);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

qTable = rlTable(obsInfo, actInfo);
qFcn   = rlQValueFunction(qTable, obsInfo, actInfo);

qAgent = rlQAgent(qFcn);
qAgent.DiscountFactor = 1;
qAgent.LearnRate      = 0.5;
qAgent.Epsilon        = 1.0;
qAgent.EpsilonDecay   = 0.02;
qAgent.EpsilonMin     = 0.01;

trainOpts = rlTrainingOptions();
trainOpts.MaxEpisodes        = 200;
trainOpts.MaxStepsPerEpisode = 10;

qStats  = train(qAgent, env, trainOpts);
reward  = sim(qAgent, env);
fprintf('MDP Q-learning cumulative reward: %.1f\n', reward);
