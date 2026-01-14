% Test 9: Valid indexing behavior, no warnings are expected
% A(1,2) is treated as a scalar.
% x = A(1,2) and y = x + 1 should both remain scalar

% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = matrix[3 x 4]
% EXPECT: C = matrix[3 x 4]
% EXPECT: x = scalar
% EXPECT: y = scalar

A = zeros(3, 4);
x = A(1, 2);
y = x + 1;
B = zeros(3, 4);
C = A + B;
