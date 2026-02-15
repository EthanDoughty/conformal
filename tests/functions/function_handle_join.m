% Test: Function handle joined with scalar → unknown
% EXPECT: warnings = 0
% EXPECT: f = unknown

if 1
    f = @(x) x;
else
    f = 5;
end
