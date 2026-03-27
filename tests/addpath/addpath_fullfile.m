% Test addpath with fullfile constant folding
% EXPECT_NO_WARNING: W_UNKNOWN_FUNCTION

addpath(fullfile('deep', 'sub1'));
r = deep_func(42);
