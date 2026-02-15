% Test: Cell literal dimensions
% EXPECT: warnings = 0
% EXPECT: c1 = cell[1 x 3]
% EXPECT: c2 = cell[2 x 2]
% EXPECT: c3 = cell[3 x 1]

c1 = {1, 2, 3};
c2 = {1, 2; 3, 4};
c3 = {1; 2; 3};
