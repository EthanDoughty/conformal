% Test: confirmed warning for filter mode testing
% --witness filter would show only this confirmed bug
% EXPECT: warnings = 1

A = zeros(3, 4);
B = zeros(5, 2);
C = A * B;
