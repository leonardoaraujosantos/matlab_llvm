% examples/instrument/tcp_sender.m
% The client end of a two-process TCP link. Run tcp_receiver.m FIRST in another
% terminal, then run this:
%     matlabc -repl < tcp_sender.m
PORT = 40123;
c = tcpclient("127.0.0.1", PORT);
fprintf('connected to 127.0.0.1:%d ; sending 5 samples ...\n', PORT);
for k = 1:5
    c.write(k * 10.0);
    fprintf('  sent %.1f\n', k * 10.0);
end
fprintf('sender done\n');
