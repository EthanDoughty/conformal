% Test: Elseif chain without else clause
% Missing else branch means A may be uninitialized (bottom → unknown via join)
% EXPECT: warnings = 0
% EXPECT: A = matrix[4 x 4]

n = 2;
if n > 10
    A = zeros(4, 4);
elseif n > 5
    A = ones(4, 4);
end
