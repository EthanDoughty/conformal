% Test 25: Parse recovery - multi-line unsupported with braces
% Newlines inside delimiters should not terminate statement
% Tests delimiter depth tracking during recovery

% EXPECT: warnings = 0
% EXPECT: A = matrix[2 x 2]
% EXPECT: foo = unknown
% EXPECT: C = matrix[2 x 2]

A = zeros(2, 2);
foo = bar{1,
          2};
C = A + A;
