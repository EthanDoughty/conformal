% Test: end-less function with inner if/end block
% EXPECT: warnings = 0
A = endless_inner_blocks_helper(zeros(3, 3), 1);
% EXPECT: A = matrix[3 x 3]
