% conformal:type x [17x1]
% conformal:type P [17x17]
% conformal:type flag scalar
% conformal:type name string
% Test: % conformal:type directives seed variable shapes before analysis
% EXPECT_NO_WARNING
% EXPECT: x = matrix[17 x 1]
% EXPECT: P = matrix[17 x 17]
% EXPECT: flag = scalar
% EXPECT: name = string

% These variables are pre-seeded; arithmetic should type-check cleanly.
P_new = P - x * x';
% EXPECT: P_new = matrix[17 x 17]
