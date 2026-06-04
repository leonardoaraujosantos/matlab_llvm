e = min([]);
fprintf('min: isempty=%.0f numel=%.0f\n', isempty(e), numel(e));
f = max([]);
fprintf('max: isempty=%.0f numel=%.0f\n', isempty(f), numel(f));
fprintf('nonempty: %.0f %.0f\n', min([3 1 2]), max([3 1 2]));
