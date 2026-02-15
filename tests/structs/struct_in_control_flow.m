% Test: Struct shapes joined in branches (union with bottom)
% If-branch has {x, y}, else has {x} → join gives {x: scalar, y: bottom→scalar}
% But bottom is filtered from display, so y appears as scalar from if-branch
% EXPECT: warnings = 0
% EXPECT: s = struct{x: scalar, y: scalar}

s.x = 1;
if 1
    s.x = 5;
    s.y = 10;
else
    s.x = 3;
end
