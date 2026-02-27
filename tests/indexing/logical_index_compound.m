A = rand(4, 4);
B = A(A > 0 & A < 1);
% EXPECT: B = matrix[None x 1]
% EXPECT: warnings = 2
