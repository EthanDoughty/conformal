% Cell assignment through a variable with known value preserves element tracking.
% EXPECT: warnings = 0
% EXPECT: x = matrix[3 x 3]
c = {1, 2, 3};
i = 2;
c{i} = zeros(3, 3);
x = c{2};
