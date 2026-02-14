% Test: Unknown function in loop + downstream matmul (false positive before fix)
% Pre-fix: fixpoint preserves matrix[3x3], causing spurious inner-dim warning on A*zeros(5,5)
% Post-fix: A = unknown after loop, no spurious warning
% EXPECT: warnings = 1
% EXPECT: A = unknown
% EXPECT: B = unknown
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = unknown
% EXPECT_FIXPOINT: B = unknown

A = zeros(3, 3);
for i = 1:n
    A = unknown_func();
end
B = A * zeros(5, 5);
