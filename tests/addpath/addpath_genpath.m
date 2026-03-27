% Test addpath with genpath (recursive scan)
% EXPECT_NO_WARNING: W_UNKNOWN_FUNCTION

addpath(genpath('deep'));
r = deep_func(42);
