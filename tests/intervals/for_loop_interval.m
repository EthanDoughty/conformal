% EXPECT: warnings = 0
A = zeros(10, 10);
for i = 1:10
    x = A(i, 1);
end
% EXPECT: A = matrix[10 x 10]
