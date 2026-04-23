disp(count());
disp(count());
disp(count());

function y = count()
    persistent n;
    n = n + 1;
    y = n;
end
