% hmm_markov.m — Statistics Toolbox Tier-6: Hidden Markov Models.
% ----------------------------------------------------------------------
% The "occasionally dishonest casino": a 2-state HMM switches between a
% fair die (uniform emissions) and a loaded die (biased toward a 6).
% Generate a sequence, recover the most likely hidden state path with the
% Viterbi algorithm, score it with forward-backward decoding, and re-learn
% the model parameters from data with Baum-Welch (`hmmtrain`).
% States 1 = fair, 2 = loaded; symbols 1..2 (here a coarse low/high die).
TRANS = [0.95 0.05;     % fair  -> stays fair / switches to loaded
         0.10 0.90];    % loaded-> back to fair / stays loaded
EMIS  = [0.5 0.5;       % fair  : low / high equally likely
         0.1 0.9];      % loaded: mostly "high"

rng(3);
[seq, states] = hmmgenerate(40, TRANS, EMIS);
fprintf('first 6 emissions : %.0f %.0f %.0f\n', seq(1), seq(2), seq(3));
fprintf('true states 1..3  : %.0f %.0f %.0f\n', states(1), states(2), states(3));

% ----- Viterbi: most likely hidden-state path -------------------------
vpath = hmmviterbi(seq, TRANS, EMIS);
fprintf('viterbi   1..3    : %.0f %.0f %.0f\n', vpath(1), vpath(2), vpath(3));

% ----- forward-backward posterior decoding ----------------------------
[pstates, logpseq] = hmmdecode(seq, TRANS, EMIS);
fprintf('log P(seq)        : %.3f\n', logpseq);
fprintf('P(loaded | t=1)   : %.3f\n', pstates(2, 1));

% ----- Baum-Welch: re-learn the model from a long sequence ------------
rng(5);
[train, tstates] = hmmgenerate(800, TRANS, EMIS);
[Test, Eest] = hmmtrain(train, [0.6 0.4; 0.4 0.6], [0.5 0.5; 0.3 0.7]);
fprintf('learned stay-fair : %.2f  (true 0.95)\n', Test(1, 1));
fprintf('learned loaded-hi : %.2f  (true 0.90)\n', Eest(2, 2));
