% Test: 2D cell indexing with variable indices resolved via getExprInterval.
% EXPECT: warnings = 0
% EXPECT: x = scalar
% EXPECT: y = matrix[2 x 3]

c = cell(2, 2);
c{1, 1} = 42;
c{1, 2} = zeros(2, 3);
c{2, 1} = 'hello';
c{2, 2} = ones(4, 4);
r = 1;
k = 2;
x = c{r, r};
y = c{r, k};
