% Test: Type Error Detection (v1.9.0)
% Verify warnings for non-numeric types in arithmetic/transpose/negation/concat

% EXPECT: warnings = 12

% Arithmetic type errors (struct, cell, function_handle)
s.x = 1;
A = s + 5;                  % struct + scalar -> W_ARITHMETIC_TYPE_MISMATCH
B = s * ones(3,3);          % struct * matrix -> W_ARITHMETIC_TYPE_MISMATCH
C = s .* 2;                 % struct .* scalar -> W_ARITHMETIC_TYPE_MISMATCH
D = s - 1;                  % struct - scalar -> W_ARITHMETIC_TYPE_MISMATCH
E = s / 2;                  % struct / scalar -> W_ARITHMETIC_TYPE_MISMATCH
c = cell(2,2);
F = c + 1;                  % cell + scalar -> W_ARITHMETIC_TYPE_MISMATCH
f = @(x) x;
G = f + 1;                  % function_handle + scalar -> W_ARITHMETIC_TYPE_MISMATCH

% Unary type errors
H = s';                     % transpose of struct -> W_TRANSPOSE_TYPE_MISMATCH
I = -s;                     % negation of struct -> W_NEGATE_TYPE_MISMATCH

% Concatenation type errors
J = [s; ones(3,3)];         % struct in concat -> W_CONCAT_TYPE_MISMATCH

% Regression tests (should NOT warn for type errors)
L = ones(3,3) + eye(3);     % numeric + numeric -> no type warning
N = 5 * ones(2,2);          % scalar * matrix -> no type warning
O = -ones(3,1);             % negation of matrix -> no type warning
P = ones(2,3)';             % transpose of matrix -> no type warning

% Unknown operand test (sound -- no type warning, only W_UNKNOWN_FUNCTION)
K = unknown_result() + 1;   % unknown + scalar -> only W_UNKNOWN_FUNCTION

% String regression (only W_STRING_ARITHMETIC, not W_ARITHMETIC_TYPE_MISMATCH)
Q = 'hello' + ones(2,2);    % only W_STRING_ARITHMETIC

% EXPECT: A = unknown
% EXPECT: B = unknown
% EXPECT: C = unknown
% EXPECT: D = unknown
% EXPECT: E = unknown
% EXPECT: F = unknown
% EXPECT: G = unknown
% EXPECT: H = unknown
% EXPECT: I = unknown
% EXPECT: J = unknown
% EXPECT: K = unknown
% EXPECT: L = matrix[3 x 3]
% EXPECT: N = matrix[2 x 2]
% EXPECT: O = matrix[3 x 1]
% EXPECT: P = matrix[3 x 2]
% EXPECT: Q = unknown
