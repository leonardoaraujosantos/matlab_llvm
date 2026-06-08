% Bluetooth Toolbox Tier-3/4 — protocol data units + channel selection.
%   bleLLDataChannelPDU / Decode, bleL2CAPFrame / Decode (byte-packing
%   round-trip), bleChannelSelection (CSA #1/#2 hop sequence) and
%   bleChannelIndexToFrequency (the 2.4 GHz channel map).
rng(4);
payload = round(rand(80, 1));            % 10-byte payload

% LL data-channel PDU gen -> decode round trip.
pdu = bleLLDataChannelPDU(2, payload);
fprintf('LL PDU bits: %.0f\n', numel(pdu));     % 16 header + 80 = 96
d = bleLLDataChannelPDUDecode(pdu);
fprintf('LL  LLID=%.0f Length=%.0f payloadBER=%.4f\n', d.LLID, d.Length, biterr(payload, d.Payload));

% L2CAP frame gen -> decode round trip.
fr = bleL2CAPFrame(64, payload);
fprintf('L2CAP bits: %.0f\n', numel(fr));        % 32 header + 80 = 112
df = bleL2CAPFrameDecode(fr);
fprintf('L2CAP CID=%.0f Length=%.0f payloadBER=%.4f\n', df.CID, df.Length, biterr(payload, df.Payload));

% Channel Selection Algorithm #1 (additive hop by 7, mod 37 data channels).
s1 = bleChannelSelection(1, 7, 6);
fprintf('CSA1 hops: %.0f %.0f %.0f %.0f %.0f %.0f\n', s1(1),s1(2),s1(3),s1(4),s1(5),s1(6));

% Channel index -> RF centre frequency (MHz).
f = bleChannelIndexToFrequency([0; 19; 39]);
fprintf('freqs MHz: %.0f %.0f %.0f\n', f(1), f(2), f(3));
