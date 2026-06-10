% struct_field_typing.m — #191 P4.2. Assigning a struct field records its type
% on a per-binding struct type (StructType.Fields, OpenSet), so a later
% `s.field` read recovers it instead of falling to Any. An unassigned field
% stays Any (the struct is an open set). Shapes are preserved through the field.

s.a = 1;
s.b = "hello";
s.c = [1 2 3];

x = s.a;        % double
y = s.b;        % string
z = s.c;        % double row vector (shape preserved)
w = s.missing;  % any (field never assigned)

% Field type flows into further inference.
x2 = s.a + 1;   % double
