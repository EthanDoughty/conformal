% Symbolic range indexing
A = zeros(5, 4);
B = zeros(n, m);

% Variable endpoint: extent = k
C = A(1:k, :);
% EXPECT: C = matrix[k x 4]

% Variable both endpoints: extent = j-k+1
D = B(k:j, :);
% EXPECT: D = matrix[(j-k+1) x m]

% Symbolic cancellation: k:k+2 -> extent 3
E = B(k:k+2, :);
% EXPECT: E = matrix[3 x m]

% EXPECT: warnings = 0
