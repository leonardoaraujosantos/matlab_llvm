% tier7_smoke.m — exercise every Tier-7 modern-codes entry once.
%
% Tier-7 ships the function-form LDPC / Turbo / Polar surface
% described in docs/comm_toolbox_roadmap.md §5.4.  The classdef
% System-Object variants stay gated on the SO lowering fix.
%
% Each section runs a small encode -> AWGN -> decode round-trip and
% reports the decoded-vs-original error count.

rng(2035);

% ============================================================
% §5.4.A Polar (N = 16, K = 8)
% ============================================================
disp('=== Polar (N=16, K=8) ===');
N = 16;
% Lower half frozen (positions 1..8); information bits in 9..16.
% For real 3GPP polar code the frozen set comes from a sub-channel
% reliability sequence; this lower-half choice is a stand-in for
% the demo.
frozen = zeros(N, 1);
for i = 1:8
    frozen(i) = 1;
end
% Build u with information bits in positions 9..16
u = zeros(N, 1);
info = randi(2, 8, 1) - 1;
for i = 1:8
    u(i + 8) = info(i);
end
disp('info bits:'); disp(info');
cw = polarEncode(u, N);
fprintf('codeword length: %.0f\n', size(cw, 1));
tx = 1 - 2 * cw;
rx = awgn(tx, 4);
llr = 2 * rx;
u_hat = polarSCdecode(llr, frozen, N);
% Extract info bits from u_hat
info_hat = zeros(8, 1);
for i = 1:8
    info_hat(i) = u_hat(i + 8);
end
fprintf('polar errors (8 info bits): %.0f\n', biterrCount(info, info_hat));

% ============================================================
% §5.4.B LDPC (6, 3) hand-rolled
% ============================================================
disp(' ');
disp('=== LDPC (6, 3) hand-rolled ===');
% Systematic: G = [I_3 | P] with P =
%   [1 1 0]
%   [0 1 1]
%   [1 0 1]
% so H = [P^T | I_3] =
%   [1 0 1 1 0 0]
%   [1 1 0 0 1 0]
%   [0 1 1 0 0 1]
P_ldpc = [1 1 0; 0 1 1; 1 0 1];
H_ldpc = [1 0 1 1 0 0; 1 1 0 0 1 0; 0 1 1 0 0 1];
msg = [1; 0; 1];
cw  = ldpcEncode(msg, P_ldpc);
fprintf('LDPC codeword: '); disp(cw');
tx = 1 - 2 * cw;
rx = awgn(tx, 5);
llr = 2 * rx;
dec = ldpcDecodeMS(llr, H_ldpc, 20);
fprintf('LDPC decoded : '); disp(dec');
fprintf('LDPC errors vs codeword: %.0f\n', biterrCount(cw, dec));

% ============================================================
% §5.4.C Turbo (PCCC with (7, 5)_8 RSC + identity permutation)
% ============================================================
disp(' ');
disp('=== Turbo PCCC (k=64) at 4 dB SNR ===');
gens = [oct2dec(7), oct2dec(5)];
t = poly2trellis(3, gens);
k = 64;
msg = randi(2, k, 1) - 1;
% Shift-by-11 permutation (non-trivial); 3GPP-style polynomial
% permutations are caller-supplied lookups for production use.
perm = zeros(k, 1);
for i = 1:k
    perm(i) = mod(i - 1 + 11, k) + 1;
end
code = turboEncode(msg, t, perm);
fprintf('turbo code length (= 3 * k): %.0f\n', size(code, 1));
tx = 1 - 2 * code;
rx = awgn(tx, 4);
llr = 2 * rx;
llr_sys = llr(1:k);
llr_p1  = llr(k+1:2*k);
llr_p2  = llr(2*k+1:3*k);
dec = turboDecode(llr_sys, llr_p1, llr_p2, t, perm, 6);
fprintf('turbo errors: %.0f\n', biterrCount(msg, dec));
