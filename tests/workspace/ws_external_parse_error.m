% Test: calling an external function with syntax errors emits W_EXTERNAL_PARSE_ERROR
% EXPECT: warnings >= 1
result = broken_helper(1); % EXPECT_WARNING: W_EXTERNAL_PARSE_ERROR
