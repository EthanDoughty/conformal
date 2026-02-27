A = rand(3, 3);
mask = A > 0.5;
B = A(mask);
% EXPECT: mask = matrix[3 x 3]
% EXPECT: B = matrix[None x 1]
