% Pentagon bridge fires inside loop body during fixpoint phases.
% n = size(A,1) creates DimEquiv alias; i <= n via Pentagon.
% In fixpoint mode the bridge re-applies each phase, keeping i's interval tight.
% EXPECT: warnings = 0
% EXPECT_FIXPOINT: warnings = 0
A = zeros(5, 5);
n = size(A, 1);
for i = 1:n
    x = A(i, 1);
end
