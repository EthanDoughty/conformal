% Test 30: Apply node disambiguation
% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = matrix[3 x 1]
% EXPECT: C = matrix[2 x 3]
% EXPECT: D = unknown
% EXPECT: E = matrix[2 x 4]
% EXPECT: M = matrix[3 x 4]
% EXPECT: val = scalar

A = [1 2 3 4; 5 6 7 8; 9 10 11 12];
B = A(:, 2);
C = zeros(2, 3);
D = my_func(5);
E = A(1:2, :);
M = zeros(3, 4);
val = M(2, 3);
