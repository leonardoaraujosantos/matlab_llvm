% UDP loopback parity (interpret vs compile): send fixed payloads on
% 127.0.0.1 and read them back. Uses a single class (udpport) for both
% endpoints so the C++ emit lane stays clean, and fixed values + same-process
% loopback so the result is deterministic. Regression guard for the
% network-io-matlab capture surface (sockets keyed by the handle, float64
% payloads, matlab_net_read typed as a Matrix in the compiled lane).
rx = udpport(51721);
tx = udpport(0);

tx.write([11 22 33], "127.0.0.1", 51721);
a = rx.read(3);
disp('a:'); disp(a);

tx.write([44 55], "127.0.0.1", 51721);
b = rx.read(2);
disp('b:'); disp(b);

disp('sum a:'); disp(sum(a));
