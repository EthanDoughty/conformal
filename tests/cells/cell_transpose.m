% Test: Transpose after curly indexing (verifies } in transpose context)
% EXPECT: warnings = 0
% EXPECT: c = cell[1 x 2]
% EXPECT: x = unknown

c = {zeros(3, 3), ones(2, 2)};
x = c{1}';
