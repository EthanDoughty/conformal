% Test: Nested for loops
% Loop variables i and j are scalar, treated as symbolic dimensions
% EXPECT: warnings = 0
% EXPECT: i = scalar
% EXPECT: j = scalar
% EXPECT: B = matrix[i x j]

for i = 1:n
    for j = 1:m
        B = zeros(i, j);
    end
end
