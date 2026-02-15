% Test: Create struct via field assignment
% EXPECT: warnings = 0
% EXPECT: s = struct{x: scalar, y: matrix[3 x 1]}

s.x = 5;
s.y = zeros(3, 1);
