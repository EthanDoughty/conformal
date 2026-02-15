% Test: Cell assignment preserves dimensions
% EXPECT: warnings = 0
% EXPECT: c = cell[2 x 3]
% EXPECT: x = unknown

c = {1, 2, 3; 4, 5, 6};
c{1, 2} = 'modified';
x = c{1, 2};
