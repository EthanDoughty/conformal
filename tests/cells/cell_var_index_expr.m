% Test: Cell read with expression index (i+1, i-1) resolved via getExprInterval.
% EXPECT: warnings = 0
% EXPECT: x = matrix[2 x 3]
% EXPECT: y = scalar

c = {42, zeros(2, 3), 'hello'};
i = 1;
x = c{i + 1};
j = 3;
y = c{j - 2};
