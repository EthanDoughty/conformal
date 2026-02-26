% Test: branch join drops equivalences established in only one branch.
% The constraint 5 == m from A*B inside the if-branch is not present in the
% else-branch, so after join the equivalence is dropped. D remains matrix[m x 3].
% EXPECT: warnings = 0
% EXPECT: D = matrix[m x 3]
A = rand(n, 5);
B = rand(m, 3);
if cond
    C = A * B;
end
D = zeros(m, 3);
