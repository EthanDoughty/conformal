% Indexed assignment inside for loop (common real-world pattern)
% EXPECT: warnings = 0

A = zeros(5, 5);
for i = 1:5
    for j = 1:5
        A(i, j) = i + j;
    end
end
% EXPECT: A = matrix[5 x 5]

% Row-wise fill
B = zeros(3, 4);
for k = 1:3
    B(k, :) = ones(1, 4);
end
% EXPECT: B = matrix[3 x 4]
