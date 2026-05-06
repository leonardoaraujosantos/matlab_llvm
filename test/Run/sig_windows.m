% Tier-1 (Signal Processing Toolbox roadmap §2.3): the windows tail.
% Each window returns a column vector of length N. Print the centre tap
% and the sum so the golden is small and lane-stable. Symmetric (non-
% periodic) form throughout.
N = 7;

% Already-shipped trio (retrofit Python+TS in this slice).
disp(sum(hamming(N)));
disp(sum(hann(N)));
disp(sum(blackman(N)));

% New tail.
disp(sum(rectwin(N)));        % == N
disp(sum(triang(N)));
disp(sum(bartlett(N)));
disp(sum(barthannwin(N)));
disp(sum(bohmanwin(N)));
disp(sum(parzenwin(N)));
disp(sum(nuttallwin(N)));
disp(sum(blackmanharris(N)));
disp(sum(flattopwin(N)));

% Two-arg parametric windows.
disp(sum(kaiser(N, 6.0)));
disp(sum(tukeywin(N, 0.5)));
disp(sum(gausswin(N, 2.5)));
disp(sum(chebwin(N, 60.0)));
disp(sum(taylorwin(N, 4, -30)));

% Centre taps (peak == 1 for windows that hit unity at the midpoint).
ham = hamming(N);
disp(ham(4));
hn = hann(N);
disp(hn(4));
rw = rectwin(N);
disp(rw(4));
tr = triang(N);
disp(tr(4));
kw = kaiser(N, 6.0);
disp(kw(4));
gw = gausswin(N, 2.5);
disp(gw(4));
bh = blackmanharris(N);
disp(bh(4));
