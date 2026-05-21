% Coverage edge case: variables in all three buckets (tracked, partial, untracked)
% EXPECT: A = matrix[2 x 3]
% EXPECT: i = scalar
% EXPECT: C = matrix[None x 3]
% EXPECT: y = unknown
A = [1 2 3; 4 5 6];
y = unknownFunc(A);  % EXPECT_WARNING: W_UNKNOWN_FUNCTION
C = zeros(3, 3);
for i = 1:n
    if cond
        C = [C; zeros(1, 3)];  % EXPECT_WARNING: W_REASSIGN_INCOMPATIBLE
    end
end
