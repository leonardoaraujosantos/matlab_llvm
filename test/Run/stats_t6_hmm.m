% Statistics Toolbox Tier-6 — Hidden Markov Models.
TRANS = [0.95 0.05; 0.10 0.90];
EMIS  = [0.5 0.5; 0.1 0.9];
rng(3);
[seq, states] = hmmgenerate(20, TRANS, EMIS);
fprintf('seq3   %.0f %.0f %.0f\n', seq(1), seq(2), seq(3));
vit = hmmviterbi(seq, TRANS, EMIS);
fprintf('vit3   %.0f %.0f %.0f\n', vit(1), vit(2), vit(3));
[ps, lp] = hmmdecode(seq, TRANS, EMIS);
fprintf('logp   %.4f\n', lp);
fprintf('post11 %.4f\n', ps(1,1));
rng(5); [longseq, lst] = hmmgenerate(500, TRANS, EMIS);
[Te, Ee] = hmmtrain(longseq, [0.6 0.4; 0.4 0.6], [0.5 0.5; 0.3 0.7]);
fprintf('T11    %.2f\n', Te(1,1));
fprintf('E22    %.2f\n', Ee(2,2));
