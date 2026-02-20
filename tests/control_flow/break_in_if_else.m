% Test: break in simple if inside loop (else branch should be skipped by break propagation)
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
A = zeros(3, 3);
for i = 1:10
    if i > 5
        break;
    else
        A = zeros(3, 3);
    end
end
