% Test: Switch/case with otherwise
% All branches assign compatible shapes
% EXPECT: warnings = 0
% EXPECT: result = matrix[2 x 2]

mode = 1;
switch mode
    case 1
        result = zeros(2, 2);
    case 2
        result = ones(2, 2);
    otherwise
        result = eye(2);
end
