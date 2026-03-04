% Test: Cell read with variable index that has known singleton interval.
% Handles bare Var, Const, and BinOp (i+1) via getExprInterval.
% EXPECT: warnings = 0
% EXPECT: x = scalar
% EXPECT: y = matrix[2 x 3]
% EXPECT: z = string

c = {42, zeros(2, 3), 'hello'};
i = 1;
x = c{i};
j = 2;
y = c{j};
k = 3;
z = c{k};
