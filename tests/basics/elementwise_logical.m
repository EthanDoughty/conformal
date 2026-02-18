% Test: element-wise logical operators & and |

% EXPECT: warnings = 0
% EXPECT: s = scalar
% EXPECT: t = scalar
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = matrix[3 x 4]
% EXPECT: C = matrix[3 x 4]
% EXPECT: D = matrix[3 x 4]

% scalar & scalar -> scalar
s = 1 & 0;

% scalar | scalar -> scalar
t = 0 | 1;

% matrix & matrix -> matrix (same shape)
A = zeros(3, 4);
B = zeros(3, 4);
C = A & B;

% matrix | matrix -> matrix (same shape)
D = A | B;
