% Test: Unknown function result overwrites variable in loop body
% v0.9.3: Fixed with Shape.bottom() — widen(matrix, unknown) now gives unknown
% (unknown is top/error, not bottom/unbound)
% EXPECT: warnings = 1
% EXPECT: A = unknown
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = unknown

A = zeros(3, 3);
for i = 1:n
    A = unknown_func();
end
