% Test: Conditional growth inside loop body
% Only one branch of if modifies A; else keeps old value
% The if-join already widens, then loop widening preserves it
% EXPECT: warnings = 1
% EXPECT: A = matrix[None x 3]
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = matrix[None x 3]

A = zeros(3, 3);
for i = 1:n
    if cond
        A = [A; zeros(1, 3)];
    end
end
