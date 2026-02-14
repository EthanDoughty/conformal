% Test: Unknown function result overwrites variable in loop body
% Known limitation: fixpoint mode loses unknown-as-error info
% because widen_shape treats unknown as bottom (unbound var semantics).
% Correct fixpoint answer would be A = unknown, but widening
% returns the pre-loop shape since widen_shape(matrix, unknown) = matrix.
% EXPECT: warnings = 1
% EXPECT: A = unknown
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = matrix[3 x 3]

A = zeros(3, 3);
for i = 1:n
    A = unknown_func();
end
