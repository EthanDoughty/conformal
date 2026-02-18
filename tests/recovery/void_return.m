% Test: function [] = name() void-return syntax
% EXPECT: warnings = 0

function [] = do_nothing(x)
    y = x + 1;
end

x = 5;
% EXPECT: x = scalar
