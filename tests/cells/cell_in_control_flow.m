% Test: Cell shapes in if/else branches
% EXPECT: warnings = 0
% EXPECT: c = cell[None x None]

if 1
    c = {1, 2; 3, 4};
else
    c = {5; 6; 7};
end
