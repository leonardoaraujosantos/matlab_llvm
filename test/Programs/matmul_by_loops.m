A = [1 2; 3 4];
B = [5 6; 7 8];
C = [0 0; 0 0];
for i = 1:2
    for j = 1:2
        sumv = 0;
        for k = 1:2
            sumv = sumv + A(i, k) * B(k, j);
        end
        C(i, j) = sumv;
    end
end
