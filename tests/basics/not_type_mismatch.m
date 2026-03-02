% Logical NOT on non-numeric type (function handle)
% EXPECT: warnings = 1
f = @sin;
result = ~f;  % EXPECT_WARNING: W_NOT_TYPE_MISMATCH
