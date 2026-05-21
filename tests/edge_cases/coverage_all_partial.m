% Coverage edge case: matrix variables with at least one Unknown dim
% Conditional growth inside loop forces widening to None on row dim.
% EXPECT: A = matrix[None x 3]
% EXPECT: B = matrix[None x 2]
A = zeros(3, 3);
B = zeros(2, 2);
for i = 1:n
    if cond
        A = [A; zeros(1, 3)];  % EXPECT_WARNING: W_REASSIGN_INCOMPATIBLE
        B = [B; zeros(1, 2)];  % EXPECT_WARNING: W_REASSIGN_INCOMPATIBLE
    end
end
