% ble_freq_hopping.m — Bluetooth Toolbox Phase-B (Tier-4).
% ----------------------------------------------------------------------
% Bluetooth LE adaptive frequency hopping: derive the per-connection-event
% data-channel sequence with bleChannelSelection (Channel Selection Algorithm
% #1), map each channel index to its 2.4 GHz RF centre frequency, and generate
% the LE waveform that would be transmitted on each hop.  Demonstrates the
% channel-selection -> channel->frequency -> per-hop waveform pipeline.

rng(1);
hopIncrement = 7;
numEvents    = 8;

% CSA #1 data-channel sequence (0..36) and its RF frequencies.
seq = bleChannelSelection(1, hopIncrement, numEvents);
fprintf('CSA#1 channel sequence (hopIncrement=%d):\n  ', hopIncrement);
for i = 1:numEvents
    fprintf('%d ', seq(i));
end
fprintf('\n');

freqs = bleChannelIndexToFrequency(seq);
fprintf('RF centre frequencies (MHz):\n  ');
for i = 1:numEvents
    fprintf('%.0f ', freqs(i));
end
fprintf('\n');

% Generate the LE waveform transmitted on the first hop's channel.
payload = round(rand(160, 1));
wf = bleWaveformGenerator(payload, 'LE1M', 8, seq(1));
rx = bleIdealReceiver(wf, 'LE1M', 8, seq(1));
fprintf('first-hop channel %d: waveform %d samples, recovered BER %.4f\n', ...
        seq(1), numel(wf), biterr(payload, rx));

% Algorithm #2 produces a pseudo-random permutation instead of a fixed step.
s2 = bleChannelSelection(2, hopIncrement, numEvents);
fprintf('CSA#2 channel sequence:\n  ');
for i = 1:numEvents
    fprintf('%d ', s2(i));
end
fprintf('\n');
