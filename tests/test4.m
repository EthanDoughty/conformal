% Test 4: Invalid matrix multiplication, dimension mismatch, warning is expected
% A is 3x4 and x is 5x1, so the inner dimensions of 4 and 5 do not match

% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 4]
% EXPECT: x = matrix[5 x 1]
% EXPECT: y = matrix[3 x 1]

A = zeros(3, 4);
x = zeros(5, 1);
y = A * x;
