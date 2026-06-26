% examples/instrument/udp_loopback.m
% --------------------------------------------------------------------
% Two simulations talking over a UDP socket — here both ends live in one
% program (loopback) so the demo is self-contained and runs anywhere. A
% "sensor" sends a stream of readings; a "logger" receives and prints them.
% This is the building block for two SEPARATE programs exchanging data: run
% the sender in one process and the receiver in another, pointed at the same
% host:port.
%
% Methods are called with DOT syntax (u.write(...), u.read(n)) — the same
% convention as the sim3d objects. Tier-1 payloads are raw float64.
%
% Run it interpreted, then compiled (single class, so it compiles cleanly):
%     matlabc -repl < udp_loopback.m
%     matlabc -emit-cpp udp_loopback.m > udp_loopback.cpp
%     c++ -std=c++20 -I runtime udp_loopback.cpp build/libMatlabRuntime.a -lm -o udp
%     ./udp

PORT = 50777;

logger = udpport(PORT);     % the consumer: bound to PORT
sensor = udpport(0);        % the producer: ephemeral local port

fprintf('streaming 5 readings over UDP 127.0.0.1:%d ...\n', PORT);

total = 0;
for k = 1:5
    reading = k * 1.5;                          % the "measurement"
    sensor.write(reading, "127.0.0.1", PORT);   % send it
    got = logger.read(1);                       % receive it
    total = total + got(1);
    fprintf('  sent %.2f  ->  received %.2f\n', reading, got(1));
end

fprintf('captured 5 readings; sum = %.2f\n', total);
