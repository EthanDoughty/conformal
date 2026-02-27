% Test: arrayfun(@(x) x^2, A) where A is matrix[3 x 4] -> matrix[3 x 4]
% EXPECT: warnings = 0
% EXPECT: result = matrix[3 x 4]

A = zeros(3, 4);
result = arrayfun(@(x) x^2, A);
