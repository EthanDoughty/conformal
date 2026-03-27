% Test addpath with nonexistent directory (should not crash, function stays unknown)

addpath('does_not_exist');
r = nonexistent_func(1); % EXPECT_WARNING: W_UNKNOWN_FUNCTION
