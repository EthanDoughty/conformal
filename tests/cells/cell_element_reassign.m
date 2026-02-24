% Cell element tracking survives through sequential assignments.
% EXPECT: warnings = 0
% EXPECT: x = matrix[2 x 2]
c = {};
c{1} = zeros(2, 2);
c{2} = ones(3, 1);
x = c{1};
