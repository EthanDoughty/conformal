% Test 24: Parse recovery - multiple assignment (unsupported destructuring)
% Target extraction should identify both A and B
% Both A and B should become unknown, downstream propagates

% EXPECT: warnings = 1
% EXPECT: A = unknown
% EXPECT: B = unknown
% EXPECT: C = matrix[2 x 2]
% EXPECT: E = unknown
% EXPECT: F = unknown

[A, B] = unsupported();
C = zeros(2, 2);
E = A + C;
F = B + 1;
