% Classic for-loop accumulator: sum of 1..10.
total = 0;
for i = 1:10
    total = total + i;
end
disp('sum(1..10) =');
disp(total);

% Nested loop: fill a 3x3 multiplication table.
T = zeros(3, 3);
for i = 1:3
    for j = 1:3
        T(i, j) = i * j;
    end
end
disp('3x3 multiplication table:');
disp(T);

% Non-unit and negative step.
disp('2:2:10 =');
for k = 2:2:10
    disp(k);
end

disp('countdown 5..1:');
for k = 5:-1:1
    disp(k);
end
