% Test: Early return in multi-output function leaves some outputs unset
% output2 never assigned before return → bottom → unknown at boundary
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
% EXPECT: B = unknown

function [output1, output2] = partial_return(x)
    output1 = x;
    return;
    output2 = x';
end

[A, B] = partial_return(zeros(3, 3));
