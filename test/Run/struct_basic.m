s.x = 5;
s.y = 7;
disp(s.x);
disp(s.y);
disp(s.x + s.y);

% Field update
s.x = s.x * 10;
disp(s.x);

% Multiple structs
p.name = 42;
q.name = 100;
disp(p.name);
disp(q.name);
disp(p.name + q.name);

% Dynamic field read
disp(s.('x'));
disp(p.('name'));
