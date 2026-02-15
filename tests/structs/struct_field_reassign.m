% Test: Reassign field with different shape
% EXPECT: warnings = 0
% EXPECT: s = struct{x: matrix[3 x 3]}

s.x = 5;
s.x = zeros(3, 3);
