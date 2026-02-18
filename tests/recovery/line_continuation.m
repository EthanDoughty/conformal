% Test: ... line continuation support
% EXPECT: warnings = 0

% Basic continuation
x = 1 + ...
    2;
% EXPECT: x = scalar

% Continuation in function args
y = zeros(3, ...  this is a comment
    4);
% EXPECT: y = matrix[3 x 4]

% Continuation inside matrix literal (prevents row break)
A = [1 2 ...
     3 4];
% EXPECT: A = matrix[1 x 4]

% Multiple continuations
B = zeros(2, ...
    3) + ...
    ones(2, 3);
% EXPECT: B = matrix[2 x 3]
