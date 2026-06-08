% ble_ll_pdu_roundtrip.m — Bluetooth Toolbox Phase-B (Tier-3).
% ----------------------------------------------------------------------
% Build Bluetooth LE protocol data units and decode them back: an LL
% data-channel PDU carrying an L2CAP frame (the typical nesting), then confirm
% the gen -> decode round trip recovers the header fields and payload.

rng(3);

% Application payload -> L2CAP frame (CID 64) -> LL data-channel PDU (LLID 2).
appData = round(rand(96, 1));                 % 12-byte application payload
l2 = bleL2CAPFrame(64, appData);
pdu = bleLLDataChannelPDU(2, l2);

fprintf('L2CAP frame bits: %d, LL PDU bits: %d\n', numel(l2), numel(pdu));

% Decode the LL PDU, then the contained L2CAP frame.
llinfo = bleLLDataChannelPDUDecode(pdu);
fprintf('LL  : LLID=%d  Length=%d bytes\n', llinfo.LLID, llinfo.Length);

l2info = bleL2CAPFrameDecode(llinfo.Payload);
fprintf('L2CAP: CID=%d  Length=%d bytes\n', l2info.CID, l2info.Length);

fprintf('application payload recovered, BER=%.4f\n', biterr(appData, l2info.Payload));
