% Test addpath resolution with function-call syntax
% EXPECT_NO_WARNING: W_UNKNOWN_FUNCTION

addpath('utils');
addpath('lib');

A = ones(3, 3);
B = ones(3, 3);
r1 = helper_add(A, B);
r2 = helper_mul(A, B);
