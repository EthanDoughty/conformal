% size() with concrete dims propagates through valueRanges and bridge.
% A has concrete 3x4. n = size(A,1) sets valueRanges[n]=[3,3].
% zeros(n,n) resolves n inline to Concrete 3.
% EXPECT: warnings = 0
% EXPECT: B = matrix[3 x 3]
A = rand(3, 4);
n = size(A, 1);
B = zeros(n, n);
