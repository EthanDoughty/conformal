% Test: ^ in complex expressions does not break recovery

% EXPECT: warnings = 0
% EXPECT: x = scalar
% EXPECT: y = scalar
% EXPECT: z = scalar

x = 2 ^ 10;
y = 3 ^ 2 ^ 1;
z = (1 + 2) ^ 3;
