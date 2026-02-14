% Test: Apply node disambiguates based on environment
% Bound variables are indexed, builtins are called
% EXPECT: warnings = 0
% EXPECT: M = matrix[3 x 4]
% EXPECT: val = scalar
% EXPECT: row = matrix[1 x 4]
% EXPECT: col = matrix[3 x 1]

M = zeros(3, 4);
val = M(2, 3);
row = M(1, :);
col = M(:, 2);
