% Test multiple addpath args in one call
% EXPECT_NO_WARNING: W_UNKNOWN_FUNCTION

addpath('utils', 'lib');
r1 = helper_add(ones(2), ones(2));
r2 = helper_mul(ones(2), ones(2));
