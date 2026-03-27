% Test that same-dir functions shadow addpath'd functions
% EXPECT_NO_WARNING: W_UNKNOWN_FUNCTION

addpath('utils');
A = ones(3,3);
B = ones(3,3);
r = helper_add(A, B);
