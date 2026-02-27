% n is in a DimEquiv class via size(). if r == 5 narrows r to [5,5],
% bridge propagates 5 into DimEquiv and valueRanges for equivalent var n.
% EXPECT: warnings = 0
% EXPECT: B = matrix[5 x 5]
A = rand(n, n);
r = size(A, 1);
if r == 5
    B = zeros(n, n);
end
