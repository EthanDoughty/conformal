% Test: for loops with optional parentheses
% MATLAB allows for(i = 1:n) ... end with parens around the loop variable

n = 3;
x = zeros(1, n);

for(i = 1:n)
    x(i) = i * 2;
end

for( j = 1:3 )
    x(j) = x(j) + 1;
end

% EXPECT: x: matrix[1 x 3]
