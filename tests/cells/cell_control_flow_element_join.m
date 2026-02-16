% EXPECT: warnings = 0
% EXPECT: c = cell[1 x 1]
% EXPECT: x = matrix[None x None]

if 1
    c = {zeros(2, 2)};
else
    c = {ones(3, 3)};
end
x = c{1};
