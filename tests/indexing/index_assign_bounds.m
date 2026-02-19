% Indexed assignment: MATLAB auto-expands arrays, so no OOB warnings on writes
% EXPECT: warnings = 0

M = zeros(3, 4);

% In bounds
M(1, 1) = 10;
M(3, 4) = 20;

% Beyond current bounds — MATLAB auto-expands (no warning)
M(5, 1) = 30;
M(1, 8) = 40;

% Auto-expand from empty
A = [];
A(3, 4) = 1;

% EXPECT: M = matrix[3 x 4]
% EXPECT: A = matrix[0 x 0]
