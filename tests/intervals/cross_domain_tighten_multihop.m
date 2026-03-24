% Test: multi-hop chain requiring Phase 1 iteration.
% A has concrete shape [4 x 4]. n = size(A,1) gets interval [4,4].
% m = n creates a second alias. tightenDomains must iterate to propagate
% through the chain: n -> DimEquiv -> m -> valueRanges -> zeros(m,2).
% EXPECT: warnings = 0
% EXPECT: B = matrix[4 x 2]
A = rand(4, 4);
n = size(A, 1);
m = n;
B = zeros(m, 2);
