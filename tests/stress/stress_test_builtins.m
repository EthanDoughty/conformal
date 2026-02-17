% Stress Test: Builtin Edge Cases
% Probes 2-arg forms, symbolic dims, boundary cases, and mismatched inputs.
% EXPECT: warnings = 1

% ==========================================================================
% Setup: concrete matrices
% ==========================================================================
A = zeros(3, 4);
B = ones(4, 5);
v_row = ones(1, 6);
v_col = zeros(4, 1);
sq = eye(3);

% ==========================================================================
% Reductions with explicit dimension argument
% ==========================================================================

% sum(A, 1) → reduce rows → matrix[1 x 4]
s1 = sum(A, 1);
% EXPECT: s1 = matrix[1 x 4]

% sum(A, 2) → reduce cols → matrix[3 x 1]
s2 = sum(A, 2);
% EXPECT: s2 = matrix[3 x 1]

% mean(A, 1) → reduce rows → matrix[1 x 4]
m1 = mean(A, 1);
% EXPECT: m1 = matrix[1 x 4]

% any/all with dim arg
an1 = any(A, 2);
al1 = all(A, 1);
% EXPECT: an1 = matrix[3 x 1]
% EXPECT: al1 = matrix[1 x 4]

% 1-arg reductions (default: reduce across rows)
s_ok = sum(A);
m_ok = mean(A);
% EXPECT: s_ok = matrix[1 x 4]
% EXPECT: m_ok = matrix[1 x 4]

% ==========================================================================
% linspace 2-arg form
% ==========================================================================
% linspace(a, b) defaults to 100 points: matrix[1 x 100]
ls2 = linspace(0, 1);
% EXPECT: ls2 = matrix[1 x 100]

% linspace 3-arg with known count
ls3 = linspace(0, 1, 50);
% EXPECT: ls3 = matrix[1 x 50]

% ==========================================================================
% min/max elementwise 2-arg
% ==========================================================================
M1 = zeros(3, 4);
M2 = ones(3, 4);
mn = min(M1, M2);
mx = max(M1, M2);
% EXPECT: mn = matrix[3 x 4]
% EXPECT: mx = matrix[3 x 4]

% min/max with mismatched shapes → unknown
M3 = zeros(3, 5);
mn_bad = min(M1, M3);
% EXPECT: mn_bad = unknown

% ==========================================================================
% diag edge cases
% ==========================================================================

% diag of row vector → square matrix from diagonal
d_row = diag(v_row);
% EXPECT: d_row = matrix[6 x 6]

% diag of column vector → square matrix from diagonal
d_col = diag(v_col);
% EXPECT: d_col = matrix[4 x 4]

% diag of square matrix → column vector (can't compute min(m,n) → None x 1)
d_sq = diag(sq);
% EXPECT: d_sq = matrix[None x 1]

% diag with 0 args → unknown (unsupported)
d0 = diag();
% EXPECT: d0 = unknown

% diag with 2 args → unknown (k-diagonal, unsupported)
d2 = diag(sq, 1);
% EXPECT: d2 = unknown

% ==========================================================================
% inv with concrete square matrix
% ==========================================================================
inv_sq = inv(sq);
% EXPECT: inv_sq = matrix[3 x 3]

% inv with non-square → unknown
inv_rect = inv(A);
% EXPECT: inv_rect = unknown

% ==========================================================================
% repmat
% ==========================================================================
rep1 = repmat(sq, 2, 3);
% EXPECT: rep1 = matrix[6 x 9]

% ==========================================================================
% kron — multiplicative dimensions
% ==========================================================================
K1 = zeros(2, 3);
K2 = ones(4, 5);
kr = kron(K1, K2);
% EXPECT: kr = matrix[8 x 15]

% ==========================================================================
% blkdiag — additive dimensions
% ==========================================================================
bd = blkdiag(K1, K2);
% EXPECT: bd = matrix[6 x 8]

% ==========================================================================
% reshape
% ==========================================================================
R = zeros(3, 4);
r1 = reshape(R, 4, 3);
% EXPECT: r1 = matrix[4 x 3]

r2 = reshape(R, 6, 2);
% EXPECT: r2 = matrix[6 x 2]

% reshape with element count mismatch → W_RESHAPE_MISMATCH
r_bad = reshape(R, 5, 5);

% ==========================================================================
% diff
% ==========================================================================

% diff of row vector → one shorter row vector
dv = diff(v_row);
% EXPECT: dv = matrix[1 x 5]

% diff of general matrix → (m-1) x n
dA = diff(A);
% EXPECT: dA = matrix[2 x 4]

% diff of column vector → one shorter column
dc = diff(v_col);
% EXPECT: dc = matrix[3 x 1]

% diff with 2 args → unknown (unsupported multi-arg)
d2a = diff(A, 2);
% EXPECT: d2a = unknown

% ==========================================================================
% cumsum/cumprod with 2 args → unknown
% ==========================================================================
cs1 = cumsum(A);
% EXPECT: cs1 = matrix[3 x 4]

cs2 = cumsum(A, 2);
% EXPECT: cs2 = unknown

cp2 = cumprod(A, 2);
% EXPECT: cp2 = unknown

% ==========================================================================
% atan2 with matching and mismatched shapes
% ==========================================================================
at_ok = atan2(sq, sq);
% EXPECT: at_ok = matrix[3 x 3]

at_bad = atan2(zeros(3, 3), zeros(3, 4));
% EXPECT: at_bad = unknown

% ==========================================================================
% mod/rem both matrices same shape
% ==========================================================================
mod_ok = mod(sq, sq);
% EXPECT: mod_ok = matrix[3 x 3]

rem_ok = rem(sq, sq);
% EXPECT: rem_ok = matrix[3 x 3]

% ==========================================================================
% size with dim argument → scalar
% ==========================================================================
sz1 = size(A, 1);
% EXPECT: sz1 = scalar

sz2 = size(A, 2);
% EXPECT: sz2 = scalar

% ==========================================================================
% Predicates always return scalar
% ==========================================================================
p1 = isscalar(A);
p2 = isempty(A);
p3 = isnumeric(A);
p4 = iscell(A);
% EXPECT: p1 = scalar
% EXPECT: p2 = scalar
% EXPECT: p3 = scalar
% EXPECT: p4 = scalar

% ==========================================================================
% length and numel
% ==========================================================================
ln = length(A);
% EXPECT: ln = scalar
nu = numel(A);
% EXPECT: nu = scalar
