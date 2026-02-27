% Test: cellfun with scalar-returning lambda -> matrix with cell dims
% EXPECT: warnings = 0
% EXPECT: result = matrix[4 x 1]

C = cell(4, 1);
result = cellfun(@(x) size(x, 1), C);
