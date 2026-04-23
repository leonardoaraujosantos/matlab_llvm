% Nested struct write + read
s.a.x = 5;
s.a.y = 7;
disp(s.a.x);
disp(s.a.y);
disp(s.a.x + s.a.y);

% Three levels deep
t.outer.inner.val = 42;
disp(t.outer.inner.val);

% Sibling nested fields
p.q.a = 1;
p.q.b = 2;
p.r.a = 100;
disp(p.q.a);
disp(p.q.b);
disp(p.r.a);
