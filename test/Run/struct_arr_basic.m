% Phase 2 — struct arrays. Build a struct array via assignment, read
% fields, iterate with for + length, and use sum/product reductions.

people(1).age = 30;
people(1).score = 95;
people(2).age = 25;
people(2).score = 80;
people(3).age = 40;
people(3).score = 70;

% Direct field reads.
disp(people(1).age);
disp(people(2).score);
disp(people(3).age);

% Shape / length introspection.
disp(length(people));
disp(numel(people));
disp(size(people, 2));

% Sum + filter via for-loop.
total = 0;
for k = 1:length(people)
    total = total + people(k).score;
end
disp(total);                    % 245

N = length(people);
out = zeros(1, N);
n = 0;
for k = 1:N
    if people(k).age > 28
        n = n + 1;
        out(n) = people(k).age;
    end
end
disp(n);                        % 2
disp(out(1));                   % 30
disp(out(2));                   % 40

% Cumulative product.
prod_score = 1;
for k = 1:length(people)
    prod_score = prod_score * people(k).score;
end
disp(prod_score);               % 532000
