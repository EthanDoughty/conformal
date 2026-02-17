% Adversarial Workspace Mega Test
% Tests cross-file error scenarios, cross-feature interactions (structs, cells,
% control flow, subfunctions, loops, builtins), domain-authentic pipelines,
% builtin shadowing, procedure handling, and unknown propagation.
% EXPECT: warnings = 4

% ==========================================================================
% SECTION 1: Domain-Authentic Pipelines (0 warnings)
% ==========================================================================

% Kalman prediction: F(4x4)*x(4x1)=4x1, F*P*F'+Q=4x4
F = zeros(4, 4);
x0 = zeros(4, 1);
P0 = zeros(4, 4);
Q = zeros(4, 4);
[xp, Pp] = ws_kalman_predict(F, x0, P0, Q);
% EXPECT: xp = matrix[4 x 1]
% EXPECT: Pp = matrix[4 x 4]

% State update: A(4x4)*x(4x1)+B(4x2)*u(2x1) = 4x1+4x1 = 4x1
A = zeros(4, 4);
B = zeros(4, 2);
u = zeros(2, 1);
x1 = ws_state_update(A, x0, B, u);
% EXPECT: x1 = matrix[4 x 1]

% Residual: b(4x1) - A(4x4)*x(4x1) = 4x1
b = zeros(4, 1);
r = ws_residual(A, x0, b);
% EXPECT: r = matrix[4 x 1]

% Covariance: X'(3x5)*X(5x3) = 3x3
X53 = zeros(5, 3);
cov1 = ws_covariance(X53);
% EXPECT: cov1 = matrix[3 x 3]

% Normalize: norm → scalar, v./scalar → same shape
v41 = zeros(4, 1);
nv = ws_normalize_cols(v41);
% EXPECT: nv = matrix[4 x 1]

% Gradient step: x - alpha*grad (scalar broadcast)
grad = zeros(4, 1);
xn = ws_gradient_step(x0, 1, grad);
% EXPECT: xn = matrix[4 x 1]

% ==========================================================================
% SECTION 2: Cross-Feature Returns (1 warning)
% ==========================================================================

% Struct return from external function
res = ws_make_result(zeros(3, 4));
d = res.data;
% EXPECT: d = matrix[3 x 4]
lb = res.label;
% EXPECT: lb = string

% Accessing non-existent field → W_STRUCT_FIELD_NOT_FOUND (WARNING 1)
bad_field = res.missing_field;

% Cell return from external function
cp = ws_make_cell_pair(zeros(2, 2), ones(3, 1));
% EXPECT: cp = cell[1 x 2]

% ==========================================================================
% SECTION 3: Complex External Bodies (0 warnings)
% ==========================================================================

% Conditional shape: square input → both branches produce same shape
cs_sq = ws_conditional_shape(eye(4));
% EXPECT: cs_sq = matrix[4 x 4]

% Conditional shape: non-square → join of 3x3 and 5x5 = None x None
cs_rect = ws_conditional_shape(zeros(3, 5));
% EXPECT: cs_rect = matrix[None x None]

% Subfunctions in external file
sf = ws_with_subfunc(zeros(3, 4));
% EXPECT: sf = matrix[3 x 4]

% Loop in external body (shape-stable, no widening needed)
lp = ws_with_loop(5);
% EXPECT: lp = matrix[1 x 5]

% Builtin chain: sum(4x3)→1x3, diag(1x3)→3x3, repmat(_,1,2)→3x6
bc = ws_builtin_chain(zeros(4, 3));
% EXPECT: bc = matrix[3 x 6]

% Accumulation in external body
ac = ws_accumulate(5);
% EXPECT: ac = matrix[1 x 3]
% EXPECT_FIXPOINT: ac = matrix[5 x 3]

% ==========================================================================
% SECTION 4: Caller-Side Error Detection (3 warnings)
% ==========================================================================

% Internal mismatch → silent unknown (body warning suppressed)
% zeros(3,4)*zeros(2,5): inner dim 4≠2 → W_INNER_DIM_MISMATCH inside body
% But external body warnings are suppressed → caller sees unknown, 0 warnings
silent = ws_return_unknown(zeros(3, 4), zeros(2, 5));
% EXPECT: silent = unknown

% Wrong arg count → W_FUNCTION_ARG_COUNT_MISMATCH (WARNING 2)
wrong_args = ws_two_args(zeros(3, 3));
% EXPECT: wrong_args = unknown

% Cross-file inner dim mismatch: ws_covariance(5x3) = 3x3, 3x3 * 5x5 → mismatch
% W_INNER_DIM_MISMATCH (WARNING 3)
cov_result = ws_covariance(X53);
bad_mul = cov_result * ones(5, 5);

% Multi-assign count mismatch: ws_kalman_predict returns 2, requesting 3
% W_MULTI_ASSIGN_COUNT_MISMATCH (WARNING 4)
[ka, kb, kc] = ws_kalman_predict(F, x0, P0, Q);

% ==========================================================================
% SECTION 5: Builtin Priority (0 warnings)
% ==========================================================================

% sum.m exists in workspace_adv/ but KNOWN_BUILTINS wins
% Builtin sum(4x3) → matrix[1 x 3]
% If shadow won, result would be matrix[99 x 99]
S43 = zeros(4, 3);
s = sum(S43);
% EXPECT: s = matrix[1 x 3]

% ==========================================================================
% SECTION 6: Procedure Handling (0 warnings)
% ==========================================================================

% External procedure (no return value) used in expression → unknown
% No W_PROCEDURE_IN_EXPR for external (that check only applies to same-file)
proc = ws_procedure_only(zeros(3, 3));
% EXPECT: proc = unknown

% ==========================================================================
% SECTION 7: Deep Domain Chains (0 warnings)
% ==========================================================================

% 4-deep pipeline: state_update → residual → normalize_cols → gradient_step
% A(4x4)*x(4x1)+B(4x2)*u(2x1) = 4x1
x_chain = ws_state_update(A, x0, B, u);
% EXPECT: x_chain = matrix[4 x 1]

% residual: b(4x1) - A(4x4)*x_chain(4x1) = 4x1
r_chain = ws_residual(A, x_chain, b);
% EXPECT: r_chain = matrix[4 x 1]

% normalize: same shape
n_chain = ws_normalize_cols(r_chain);
% EXPECT: n_chain = matrix[4 x 1]

% gradient_step: x_chain(4x1) - alpha*n_chain(4x1) = 4x1
result_chain = ws_gradient_step(x_chain, 0.01, n_chain);
% EXPECT: result_chain = matrix[4 x 1]

% ==========================================================================
% SECTION 8: Diamond Pattern (0 warnings)
% ==========================================================================

% Kalman predict feeds both residual (left arm) and covariance (right arm)
[xp2, Pp2] = ws_kalman_predict(F, x0, P0, Q);
% EXPECT: xp2 = matrix[4 x 1]
% EXPECT: Pp2 = matrix[4 x 4]

% Left arm: residual with predicted state
r_left = ws_residual(A, xp2, b);
% EXPECT: r_left = matrix[4 x 1]

% Right arm: covariance of predicted covariance (4x4→4x4)
cov_right = ws_covariance(Pp2);
% EXPECT: cov_right = matrix[4 x 4]

% Combine: r_left (4x1) + cov_right (4x4) * xp2 (4x1) = 4x1+4x1 = 4x1
combined = r_left + cov_right * xp2;
% EXPECT: combined = matrix[4 x 1]

% ==========================================================================
% SECTION 9: Polymorphic Stress (0 warnings)
% ==========================================================================

% ws_covariance with 5 different shapes → 5 cache entries
pc1 = ws_covariance(zeros(3, 2));
% EXPECT: pc1 = matrix[2 x 2]
pc2 = ws_covariance(zeros(4, 3));
% EXPECT: pc2 = matrix[3 x 3]
pc3 = ws_covariance(zeros(10, 5));
% EXPECT: pc3 = matrix[5 x 5]
pc4 = ws_covariance(zeros(1, 7));
% EXPECT: pc4 = matrix[7 x 7]
pc5 = ws_covariance(eye(6));
% EXPECT: pc5 = matrix[6 x 6]

% ws_kalman_predict with 3 different sizes → 3 cache entries
[pk1x, pk1P] = ws_kalman_predict(eye(3), zeros(3,1), eye(3), eye(3));
% EXPECT: pk1x = matrix[3 x 1]
% EXPECT: pk1P = matrix[3 x 3]
[pk2x, pk2P] = ws_kalman_predict(eye(5), zeros(5,1), eye(5), eye(5));
% EXPECT: pk2x = matrix[5 x 1]
% EXPECT: pk2P = matrix[5 x 5]
[pk3x, pk3P] = ws_kalman_predict(eye(2), zeros(2,1), eye(2), eye(2));
% EXPECT: pk3x = matrix[2 x 1]
% EXPECT: pk3P = matrix[2 x 2]

% ==========================================================================
% SECTION 10: Unknown Propagation (0 warnings)
% ==========================================================================

% ws_return_unknown result is unknown → operations on unknown return unknown
unk = ws_return_unknown(zeros(3, 4), zeros(2, 5));
% EXPECT: unk = unknown

% Addition with unknown → unknown
unk_add = unk + zeros(3, 3);
% EXPECT: unk_add = unknown

% Multiplication with unknown → unknown
unk_mul = unk * zeros(3, 3);
% EXPECT: unk_mul = unknown
