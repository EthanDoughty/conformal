% Test: If-elseif-else chain with compatible branches
% All branches assign matrix[3 x 3], join preserves shape
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]

n = 5;
if n < 3
    A = zeros(3, 3);
elseif n < 6
    A = ones(3, 3);
elseif n < 10
    A = eye(3);
else
    A = rand(3, 3);
end
