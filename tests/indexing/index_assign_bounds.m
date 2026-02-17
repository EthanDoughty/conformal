% Indexed assignment bounds checking
% EXPECT: warnings = 2

M = zeros(3, 4);

% In bounds: no warning
M(1, 1) = 10;
M(3, 4) = 20;

% Out of bounds row — WARNING 1
M(5, 1) = 30;

% Out of bounds column — WARNING 2
M(1, 8) = 40;

% EXPECT: M = matrix[3 x 4]
