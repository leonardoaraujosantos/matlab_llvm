% examples/instrument/tcp_receiver.m
% The server end of a two-process TCP link. Start this FIRST (it binds the
% port), then run tcp_sender.m in another terminal.
%     matlabc -repl < tcp_receiver.m
PORT = 40123;
s = tcpserver("127.0.0.1", PORT);
fprintf('listening on 127.0.0.1:%d ...\n', PORT);
for k = 1:5
    v = s.read(1);          % blocks up to the read timeout, then returns
    if isempty(v)
        fprintf('  (no data yet)\n');
    else
        fprintf('  received %.3f\n', v(1));
    end
end
fprintf('receiver done\n');
