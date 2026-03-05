% Test: [V, D] = feval('eig', A) multi-return dispatch
% EXPECT: warnings = 0

A = rand(3, 3);
[V, D] = feval('eig', A);
% EXPECT: V = matrix[3 x 3]
% EXPECT: D = matrix[3 x 3]
