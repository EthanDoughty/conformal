% EXPECT: warnings = 0
% EXPECT: t1 = matrix[3 x 3]
% EXPECT: t2 = matrix[2 x 4]
% EXPECT: f1 = matrix[n x n]
% EXPECT: n1 = matrix[3 x 5]
% EXPECT: i1 = matrix[m x n]

t1 = true(3);
t2 = true(2, 4);
f1 = false(n);
n1 = nan(3, 5);
i1 = inf(m, n);
