% Test 26: DOT token does not break elementwise ops or float literals
% Tests that adding DOT token for struct access didn't break .* ./ or floats

% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = matrix[3 x 4]
% EXPECT: C = matrix[3 x 4]
% EXPECT: D = scalar

A = zeros(3, 4);
B = A .* A;
C = A ./ A;
D = 3.14;
