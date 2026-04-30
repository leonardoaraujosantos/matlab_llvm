% Mixed string/number concatenation patterns. Three idioms covered:
%   1. `"..." + scalar + "..."`  (string operator+)
%   2. `sprintf("...%.2f...", v)` (format-style)
%   3. `['...', num2str(v), '...']` (bracket char-array concat)

% --- 1. string + scalar --------------------------------------------
idade = 25;
texto = "Eu tenho " + idade + " anos.";
disp(texto);

n = 3;
disp("hello" + n);
disp(n + " bottles");
disp("a=" + 1 + ", b=" + 2);
disp("foo" + "bar");

% Result of `+` is a real string binding.
s = "value: " + 99;
disp(s);
disp(strlen(s));
disp(isstring(s));

% --- 2. sprintf("...%.2f...") --------------------------------------
peso = 70.556;
disp(sprintf("Meu peso e %.2f kg", peso));

% --- 3. bracket char-array concat ----------------------------------
valor = 100;
disp(['O preco e ', num2str(valor), ' reais']);
disp(['x=', num2str(valor)]);
disp(['a', 'b', 'c']);
disp([num2str(1), '+', num2str(2), '=', num2str(3)]);
disp(['hi ', "world"]);

% Single-quoted char-array binding still disp's via the disp_str path.
c = 'plain char';
disp(c);
