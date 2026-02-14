% Test: Break in nested loop (only exits inner loop)
% Outer loop continues after inner break
% EXPECT: warnings = 0
% EXPECT: C = matrix[3 x 3]
% EXPECT: i = scalar
% EXPECT: j = scalar

C = zeros(3, 3);
for i = 1:5
    for j = 1:10
        if j > 3
            break;
        end
        C = eye(3);
    end
end
