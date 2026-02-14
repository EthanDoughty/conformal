% Test: Try/catch with no error
% Both try and catch branches analyzed, joined
% EXPECT: warnings = 0
% EXPECT: Y = matrix[5 x 5]

try
    Y = zeros(5, 5);
catch
    Y = ones(5, 5);
end
