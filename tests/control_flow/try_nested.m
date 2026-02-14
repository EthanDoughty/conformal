% Test: Nested try/catch blocks
% Inner try/catch joins, outer try/catch joins result
% EXPECT: warnings = 0
% EXPECT: Z = matrix[2 x 2]

try
    try
        Z = zeros(2, 2);
    catch
        Z = eye(2);
    end
catch
    Z = ones(2, 2);
end
