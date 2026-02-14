% Test: Switch without otherwise clause
% Missing otherwise means result may be uninitialized
% EXPECT: warnings = 0
% EXPECT: output = matrix[4 x 4]

val = 10;
switch val
    case 1
        output = zeros(4, 4);
    case 2
        output = ones(4, 4);
end
