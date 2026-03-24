% Test: simple Interval -> DimEquiv -> Shape chain via tightenDomains.
% A has concrete shape [5 x 3]. n = size(A,1) gets interval [5,5].
% tightenDomains propagates 5 into DimEquiv, B = zeros(n,3) resolves to [5 x 3].
% EXPECT: warnings = 0
% EXPECT: A = matrix[5 x 3]
% EXPECT: B = matrix[5 x 3]
A = rand(5, 3);
n = size(A, 1);
B = zeros(n, 3);
