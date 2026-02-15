% Test: Cell content indexing
% EXPECT: warnings = 0
% EXPECT: c = cell[2 x 2]
% EXPECT: x = unknown

A = zeros(3, 3);
B = ones(2, 2);
c = {A, B; B, A};
x = c{1, 2};
