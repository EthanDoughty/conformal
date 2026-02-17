% End on symbolic matrices
A = zeros(n, m);

% end in row position -> n
B = A(1:end, :);
% EXPECT: B = matrix[n x m]

% end-1 in row position -> n-1
C = A(1:end-1, :);
% EXPECT: C = matrix[(n-1) x m]

% end in col position -> m
D = A(:, 1:end);
% EXPECT: D = matrix[n x m]

% end-1 in col position -> m-1
E = A(:, 1:end-1);
% EXPECT: E = matrix[n x (m-1)]

% EXPECT: warnings = 0
