% Test: Cell element assignment (2D)
% EXPECT: warnings = 0
% EXPECT: c = cell[2 x 2]

c = cell(2, 2);
c{1, 1} = zeros(3, 3);
c{2, 2} = 'hello';
