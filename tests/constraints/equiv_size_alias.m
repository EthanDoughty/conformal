% Test: size() dimension aliasing records DimEquiv entries.
% r = size(A, 1) unions r with A's row dim (Concrete 3) in DimEquiv.
% c = size(A, 2) unions c with A's col dim (Concrete 4) in DimEquiv.
% zeros(r, c) uses symbolic dims r and c (size() vars remain symbolic at call site).
% EXPECT: warnings = 0
% EXPECT: B = matrix[r x c]
A = rand(3, 4);
r = size(A, 1);
c = size(A, 2);
B = zeros(r, c);
