% Test: Switch cases with shape conflicts
% Different case shapes → join loses precision
% EXPECT: warnings = 0
% EXPECT: M = matrix[None x None]

choice = 3;
switch choice
    case 1
        M = zeros(3, 3);
    case 2
        M = zeros(4, 4);
    case 3
        M = zeros(5, 5);
    otherwise
        M = zeros(6, 6);
end
