% Test: Loop variable is bound to scalar
% Loop variable i is treated as symbolic dimension (same as other scalar vars)
% EXPECT: warnings = 0
% EXPECT: i = scalar
% EXPECT: A = matrix[i x 5]

for i = 1:10
    A = zeros(i, 5);
end
