% Test: break in try/catch inside loop -- try's break takes priority
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
A = zeros(3, 3);
for i = 1:10
    try
        A = zeros(3, 3);
        break;
    catch
        A = zeros(3, 3);
        break;
    end
end
