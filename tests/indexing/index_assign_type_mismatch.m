% Indexed assignment to non-indexable type (function handle)
% EXPECT: warnings = 1
f = @sin;
f(1) = 5;  % EXPECT_WARNING: W_INDEX_ASSIGN_TYPE_MISMATCH
