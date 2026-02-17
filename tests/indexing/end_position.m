% End resolves to correct dimension per position (non-square matrix)
A = zeros(3, 7);

% Row position: end = 3
B = A(1:end, :);
% EXPECT: B = matrix[3 x 7]

% Column position: end = 7 (NOT 3)
C = A(:, 1:end);
% EXPECT: C = matrix[3 x 7]

% Column end-1: should be 6 (NOT 2)
D = A(:, 1:end-1);
% EXPECT: D = matrix[3 x 6]

% Row end-1: should be 2
E = A(1:end-1, :);
% EXPECT: E = matrix[2 x 7]

% EXPECT: warnings = 0
