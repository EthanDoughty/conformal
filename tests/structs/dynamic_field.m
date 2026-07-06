% Dynamic field access: the field name is unknowable statically, so reads stay
% conservative, the struct keeps its declared fields, and accumulation through
% a dynamic read degrades cleanly instead of warning.

s.a = 1;
names = {'a', 'b', 'c'};
name = 'a';
x = s.(name);
q = [];
for k = 1:3
    q = [q; s.(names{k})];
end

% EXPECT: warnings = 0
% EXPECT: x = unknown
% EXPECT: s = struct{a: scalar}
