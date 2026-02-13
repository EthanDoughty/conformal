% Test: Warning deduplication (dimension mismatch emitted once, not 3 times)
% EXPECT: warnings = 1
% EXPECT: A = unknown
% EXPECT: B = matrix[2 x 2]

B = zeros(2, 2);
for i = 1:3
    A = B * zeros(5, 3);
end
