% Test: Function with multiple return values
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = matrix[4 x 3]
% EXPECT: C = matrix[3 x 4]

function [out1, out2] = transpose_pair(in1)
    out1 = in1;
    out2 = in1';
end

[A, B] = transpose_pair(zeros(3, 4));
C = transpose_pair(zeros(3, 4));  % Only use first output (implicitly)
