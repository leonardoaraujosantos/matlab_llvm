% Both arms pure single-store: should collapse to `return cond ? a : b;`
% via IfStoreToSelect + Mem2RegLite.
disp(sign_of(5));
disp(sign_of(-2));

function y = sign_of(x)
    if x > 0
        y = 1;
    else
        y = -1;
    end
end
