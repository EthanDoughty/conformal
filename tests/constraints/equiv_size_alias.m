% Test: size() dimension aliasing records DimEquiv entries.
% r = size(A, 1) unions r with A's row dim (Concrete 3) in DimEquiv.
% c = size(A, 2) unions c with A's col dim (Concrete 4) in DimEquiv.
% With valueRanges set, r and c resolve inline to concrete values.
% EXPECT: warnings = 0
% EXPECT: B = matrix[3 x 4]
A = rand(3, 4);
r = size(A, 1);
c = size(A, 2);
B = zeros(r, c);
