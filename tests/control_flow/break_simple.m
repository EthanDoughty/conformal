% Test: Break statement in for loop
% Loop body may not complete all iterations
% EXPECT: warnings = 0
% EXPECT: A = matrix[10 x 10]
% EXPECT: i = scalar

A = zeros(10, 10);
for i = 1:100
    if i > 10
        break;
    end
    A = eye(10);
end
