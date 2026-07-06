% Nested block comments: the inner %{ %} pair must not close the outer block.
% If nesting were broken, the second assignment would execute and x would be 7x7.

x = ones(2, 2);
%{
%{
x = ones(5, 5);
%}
x = ones(7, 7);
%}
y = x * ones(2, 1);

% EXPECT: warnings = 0
% EXPECT: x = matrix[2 x 2]
% EXPECT: y = matrix[2 x 1]
