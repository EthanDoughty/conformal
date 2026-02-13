% Test: Variables outside loop are unaffected by loop body
% EXPECT: warnings = 0
% EXPECT: X = matrix[2 x 2]
% EXPECT: Y = matrix[3 x 3]

X = zeros(2, 2);
for i = 1:5
    Y = ones(3, 3);
end
