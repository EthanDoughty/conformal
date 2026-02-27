A = rand(5, 3);
B = A(A > 0);
% EXPECT: B = matrix[None x 1]
% EXPECT: warnings = 1
