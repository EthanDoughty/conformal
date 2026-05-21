% Coverage edge case: all variables untracked (0% coverage)
% EXPECT: x = unknown
% EXPECT: y = unknown
% EXPECT: z = unknown
x = unknownFunc1(1);  % EXPECT_WARNING: W_UNKNOWN_FUNCTION
y = unknownFunc2(2);  % EXPECT_WARNING: W_UNKNOWN_FUNCTION
z = unknownFunc3(3);  % EXPECT_WARNING: W_UNKNOWN_FUNCTION
