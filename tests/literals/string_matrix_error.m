% Test: String-matrix arithmetic produces warning
% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 3]
% EXPECT: r = unknown

A = zeros(3, 3);
r = A + 'error';
