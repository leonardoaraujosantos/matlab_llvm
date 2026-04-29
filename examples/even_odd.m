% if/else inside a for loop — classify each number in 1..N as even
% or odd. The canonical shape for branching control flow inside an
% iterative accumulator.
N = 6;
even_count = 0;
odd_count = 0;
for i = 1:N
    if mod(i, 2) == 0
        fprintf('%g is even\n', i);
        even_count = even_count + 1;
    else
        fprintf('%g is odd\n', i);
        odd_count = odd_count + 1;
    end
end
disp('even count:');
disp(even_count);
disp('odd count:');
disp(odd_count);
