% Test: For loop with symbolic range
% Loop variable i is scalar, treated as symbolic dimension
% EXPECT: warnings = 0
% EXPECT: i = scalar
% EXPECT: B = matrix[2 x i]

for i = 1:n
    B = zeros(2, i);
end
