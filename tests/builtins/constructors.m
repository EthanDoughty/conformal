% Test 31: Rich builtin shape rules
% EXPECT: warnings = 0
% EXPECT: r0 = scalar
% EXPECT: r1_concrete = matrix[3 x 3]
% EXPECT: r1_symbolic = matrix[5 x 5]
% EXPECT: r2_concrete = matrix[2 x 3]
% EXPECT: r2_symbolic = matrix[7 x 3]
% EXPECT: e0 = scalar
% EXPECT: e1_concrete = matrix[2 x 3]
% EXPECT: e1_symbolic = matrix[5 x 3]
% EXPECT: t0 = scalar
% EXPECT: t1_concrete = matrix[3 x 2]
% EXPECT: t1_symbolic = matrix[3 x 5]
% EXPECT: q0 = scalar
% EXPECT: q1_concrete = scalar
% EXPECT: q1_symbolic = scalar
% EXPECT: eye_0 = matrix[0 x 0]
% EXPECT: eye_concrete = matrix[3 x 3]
% EXPECT: eye_concrete_2arg = matrix[2 x 4]
% EXPECT: eye_symbolic = matrix[5 x 5]
% EXPECT: eye_symbolic_2arg = matrix[3 x 5]
% EXPECT: randn_0_arg = scalar
% EXPECT: randn_1_concrete = matrix[4 x 4]
% EXPECT: randn_1_symbolic = matrix[p x p]
% EXPECT: randn_2_concrete = matrix[3 x 5]
% EXPECT: randn_2_symbolic = matrix[7 x 2]
% EXPECT: abs_scalar = scalar
% EXPECT: abs_concrete = matrix[2 x 3]
% EXPECT: abs_symbolic = matrix[5 x 3]
% EXPECT: sqrt_scalar = scalar
% EXPECT: sqrt_concrete = matrix[1 x 4]
% EXPECT: sqrt_symbolic = matrix[p x q]
% EXPECT: transpose_scalar = scalar
% EXPECT: transpose_concrete = matrix[3 x 2]
% EXPECT: transpose_symbolic = matrix[3 x 5]
% EXPECT: length_concrete = scalar
% EXPECT: length_symbolic = scalar
% EXPECT: numel_concrete = scalar
% EXPECT: numel_symbolic = scalar

% Constructor tests (0-arg form: should return scalar)
r0 = randn();

% Constructor tests (1-arg form)
r1_concrete = eye(3);
n = 5;
r1_symbolic = eye(n);

% Constructor tests (2-arg form)
r2_concrete = rand(2, 3);
k = 7;
m = 3;
r2_symbolic = randn(k, m);

% Element-wise operations: abs (pass-through shape)
e0 = abs(5);
e1_concrete = abs(zeros(2, 3));
e1_symbolic = abs(ones(n, m));

% Transpose function: swap dimensions
t0 = transpose(7);
t1_concrete = transpose(zeros(2, 3));
t1_symbolic = transpose(eye(n, m));

% Query functions: length, numel (return scalar)
q0 = length([1 2 3 4 5]);
q1_concrete = numel(zeros(2, 3));
q1_symbolic = length(ones(n, m));

% Edge case: eye(0) should produce matrix[0 x 0]
eye_0 = eye(0);

% More comprehensive tests
eye_concrete = eye(3);
eye_concrete_2arg = eye(2, 4);
eye_symbolic = eye(n);
eye_symbolic_2arg = eye(m, n);

% randn with different arg counts
randn_0_arg = randn();
randn_1_concrete = randn(4);
randn_1_symbolic = randn(p);
randn_2_concrete = randn(3, 5);
randn_2_symbolic = randn(k, 2);

% abs with different input shapes
abs_scalar = abs(42);
abs_concrete = abs(zeros(2, 3));
abs_symbolic = abs(ones(n, m));

% sqrt with different input shapes
sqrt_scalar = sqrt(16);
sqrt_concrete = sqrt(rand(1, 4));
sqrt_symbolic = sqrt(eye(p, q));

% transpose with different input shapes
transpose_scalar = transpose(3.14);
transpose_concrete = transpose(zeros(2, 3));
transpose_symbolic = transpose(rand(n, m));

% length/numel with different input shapes
length_concrete = length(1:10);
length_symbolic = length(zeros(1, n));
numel_concrete = numel(ones(2, 3));
numel_symbolic = numel(randn(n, m));
