% Test: Function called with too many arguments still warns
% EXPECT: warnings = 1
% EXPECT: result = unknown

function y = one_arg(a)
    y = a * 2;
end

result = one_arg(1, 2, 3);
