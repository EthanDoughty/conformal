% Test: Phase 3 builtin shape rules
% EXPECT: warnings = 0

% --- det ---
A1 = eye(3);
det_concrete = det(A1);
% EXPECT: det_concrete = scalar
det_symbolic = det(zeros(n, n));
% EXPECT: det_symbolic = scalar

% --- diag (vector → diagonal matrix) ---
v1 = zeros(5, 1);
diag_vec_concrete = diag(v1);
% EXPECT: diag_vec_concrete = matrix[5 x 5]
v2 = zeros(n, 1);
diag_vec_symbolic = diag(v2);
% EXPECT: diag_vec_symbolic = matrix[n x n]
v3 = zeros(1, k);
diag_row_vec = diag(v3);
% EXPECT: diag_row_vec = matrix[k x k]

% --- diag (matrix → column vector) ---
M1 = zeros(3, 4);
diag_mat = diag(M1);
% EXPECT: diag_mat = matrix[None x 1]

% --- inv ---
inv_concrete = inv(eye(3));
% EXPECT: inv_concrete = matrix[3 x 3]
inv_symbolic = inv(zeros(n, n));
% EXPECT: inv_symbolic = matrix[n x n]

% --- linspace ---
ls_default = linspace(0, 1);
% EXPECT: ls_default = matrix[1 x 100]
ls_concrete = linspace(0, 10, 50);
% EXPECT: ls_concrete = matrix[1 x 50]
ls_symbolic = linspace(0, 1, n);
% EXPECT: ls_symbolic = matrix[1 x n]

% --- norm ---
norm_vec = norm(v1);
% EXPECT: norm_vec = scalar
norm_mat = norm(A1);
% EXPECT: norm_mat = scalar

% --- zeros/ones 1-arg form (gap fix) ---
z_sq = zeros(4);
% EXPECT: z_sq = matrix[4 x 4]
z_sq_sym = zeros(n);
% EXPECT: z_sq_sym = matrix[n x n]
o_sq = ones(3);
% EXPECT: o_sq = matrix[3 x 3]
o_sq_sym = ones(n);
% EXPECT: o_sq_sym = matrix[n x n]
