% source_bits.m — bit / symbol source primitives, the MATLAB-canonical
% "random data + bit-packed integer" demo. Covers:
%
%   - rng(seed) for deterministic runs
%   - randi for uniform integer sources
%   - randsrc for sampling from a custom alphabet
%   - randsrcWeighted for non-uniform alphabets
%   - randerr for crafted bit-error patterns
%   - int2bit / bit2int round-trip
%   - de2bi / bi2de LSB-first legacy round-trip

rng(0);   % default seed (== rngDefault but seeded to a deterministic state)

% --- Uniform integer source: 12 symbols from a 4-PSK constellation.
% Map an integer in {0..3} to a phase in {0, pi/2, pi, 3*pi/2}.
M = 4;
data = randi(M, 12, 1) - 1;         % {0, 1, 2, 3}
phase = (pi / 2) * data;
% Quadrature symbols: re + j*im (constructed via cos/sin).
re = cos(phase);
im = sin(phase);
disp('--- 4-PSK symbol stream ---');
fprintf('integers  : '); disp(data');
fprintf('re        : '); disp(re');
fprintf('im        : '); disp(im');

% --- Custom alphabet via randsrc: 8-ASK-style irregular constellation.
disp('--- randsrc on irregular alphabet ---');
alpha = [-7; -3; -1; 1; 3; 7];
S = randsrc(2, 8, alpha);
disp(S);

% --- Weighted alphabet: 70% zeros, 30% ones (sparse pattern).
disp('--- randsrcWeighted with skew ---');
W = randsrcWeighted(3, 10, [0; 1], [0.7; 0.3]);
disp(W);

% --- Crafted error vector: exactly 2 errors in each 8-bit row.
disp('--- randerr(4, 8, 2) ---');
E = randerr(4, 8, 2);
disp(E);

% --- int2bit / bit2int round-trip on a byte-sized alphabet.
disp('--- int2bit / bit2int round-trip ---');
sym_ints = [0; 1; 5; 7; 8; 15];
bits = int2bit(sym_ints, 4);            % 4 bits per symbol, MSB-first
disp('flat bit vector:');
disp(bits');
back = bit2int(bits, 4);
fprintf('recovered : '); disp(back');

% --- de2bi / bi2de legacy LSB-first (matches the deprecated MATLAB form).
disp('--- de2bi / bi2de round-trip ---');
b = de2bi(sym_ints, 4);
disp(b);
d = bi2de(b);
fprintf('recovered : '); disp(d');
