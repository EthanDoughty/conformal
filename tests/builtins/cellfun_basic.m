% Test: cellfun(@length, C) where C is cell(3,1) -> matrix[3 x 1]
% EXPECT: warnings = 0
% EXPECT: result = matrix[3 x 1]

C = cell(3, 1);
result = cellfun(@length, C);
