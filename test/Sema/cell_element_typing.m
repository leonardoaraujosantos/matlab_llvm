% cell_element_typing.m — #191 P4.3. A cell literal whose elements share a
% type carries that element type (CellType.ElementUpperBound), so a brace-index
% `c{i}` recovers it instead of falling to Any. A heterogeneous literal joins
% to Any and a brace-index stays untyped (see any_fallthroughs).

% Homogeneous numeric cell -> element type double.
nums = {1, 2, 3};
n = nums{2};

% Homogeneous string cell -> element type string.
strs = {"alpha", "beta"};
s = strs{1};

% Element type flows into further inference: n is double, so n + 1 is double.
m = n + 1;
