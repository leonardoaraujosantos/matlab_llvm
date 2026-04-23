% is_old(age) returns a logical: true when age > 18, else false.
disp('is_old(10):');
disp(is_old(10));
disp('is_old(18):');
disp(is_old(18));
disp('is_old(25):');
disp(is_old(25));

function r = is_old(age)
    r = age > 18;
end
