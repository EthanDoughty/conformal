% Test: Kalman gain via right division, K = P * H' / S

% EXPECT: warnings = 0
% EXPECT: K = matrix[4 x 2]
% EXPECT: x = matrix[4 x 1]

P = zeros(4, 4);
H = zeros(2, 4);
S = zeros(2, 2);
K = P * H' / S;

z = zeros(2, 1);
x = zeros(4, 1);
x = x + K * z;
