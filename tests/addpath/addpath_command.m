% Test command-syntax addpath
% EXPECT_NO_WARNING: W_UNKNOWN_FUNCTION

addpath utils
r = helper_add(ones(2,2), ones(2,2));
