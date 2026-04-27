% Phase 5.6 Stage F SV — persistent fi-array shift register.
%
% Composes Stages C (zeros init) + D (loop-iv indexing, post-
% unroll constant indices) + E (concat with static shapes) into
% the canonical FIR-style shift register:
%
%   persistent buf;
%   if isempty(buf)
%       buf = fi(zeros(1, N), S, W, F);     % Stage C reset init
%   end
%   buf = [<scalar>, buf(1:N-1)];           % Stage E concat write
%   ... = buf(k) ...                        % Stage D const reads
%
% Stage F's `LowerPersistentFiArrays` rewrites the
% `_persistent_isempty/_get_ptr/_set_ptr` runtime ABI into N
% parallel scalar persistents (synthetic indices `idx*100 + k`),
% which the existing `HWStateInfer` recognition + SV emitter
% render as N independent always_ff registers. The shift-register
% topology emerges from the per-element rewrite: each
% `buf_k_next` gets the value of the previous element / new
% input.
%
% Output for N=4 with input `x` (16-bit signed):
%   buf0_0_next = x;                        % new sample at head
%   buf0_1_next = buf0_0;                   % shift right
%   buf0_2_next = buf0_1;
%   buf0_3_next = buf0_2;
T = numerictype(1, 16, 0);
y = persistent_shift(fi(7, T));
disp(y);

function r = persistent_shift(x)
    %#codegen
    persistent buf;
    if isempty(buf)
        buf = fi(zeros(1, 4), 1, 16, 0);
    end
    buf = [x, buf(1:3)];
    r = buf(1) + buf(2) + buf(3) + buf(4);
end
