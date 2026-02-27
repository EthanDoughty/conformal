% Switch var r is in a DimEquiv class via size(). case 3 narrows r to [3,3],
% bridge propagates 3 into DimEquiv and valueRanges for equivalent var n,
% so zeros(n, n) resolves to matrix[3 x 3] inside the case body.
% EXPECT: warnings = 0
% EXPECT: B = matrix[3 x 3]
A = rand(n, n);
r = size(A, 1);
switch r
    case 3
        B = zeros(n, n);
end
