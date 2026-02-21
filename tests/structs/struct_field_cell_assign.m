% Test: cell indexed assignment into struct field
% EXPECT: s = struct{c: cell[3 x 1], name: string}
% EXPECT: warnings = 0

s.c = cell(3, 1);
s.name = 'test';
s.c{1} = 42;
s.c{2} = 'hello';
