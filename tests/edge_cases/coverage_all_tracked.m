% Coverage edge case: all variables fully tracked (100% coverage)
% EXPECT: A = matrix[2 x 3]
% EXPECT: B = matrix[3 x 2]
% EXPECT: C = matrix[2 x 2]
% EXPECT: s = scalar
A = [1 2 3; 4 5 6];
B = [1 2; 3 4; 5 6];
C = A * B;
s = 42;
