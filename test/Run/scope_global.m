global counter;
counter = 0;
bump();
bump();
bump();
disp(counter);

function bump()
    global counter;
    counter = counter + 1;
end
