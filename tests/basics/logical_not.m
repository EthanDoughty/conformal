% Test logical NOT operator (~x)
% EXPECT: warnings = 0

x = 5;
y = ~x;
% EXPECT: y = scalar

A = zeros(3, 4);
B = ~A;
% EXPECT: B = matrix[3 x 4]

% Logical NOT of boolean expression (scalar result)
cond = (x > 0);
neg_cond = ~cond;
% EXPECT: neg_cond = scalar
