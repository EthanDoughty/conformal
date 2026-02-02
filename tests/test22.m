% Test 22: Parse recovery - struct field access (unsupported)
% The analyzer should recover from unsupported struct access and continue
% B should become unknown, A should remain known

% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = unknown
% EXPECT: C = matrix[3 x 4]

A = zeros(3, 4);
B = A.field;
C = A + A;
