% Stress Test: Exhaustive analyzer exercise
% Exercises all warning codes, builtins, shape kinds, control flow, and edge cases.
% Purpose: performance benchmarking and stability testing.
%
% This file intentionally contains many errors to trigger every warning path.
% EXPECT: warnings = 36

% ==========================================================================
% SECTION 1: Shape kinds — all 7 types
% ==========================================================================

% Scalar
a_scalar = 5;
b_scalar = 3.14;

% Matrix
A = zeros(3, 4);
B = ones(4, 5);
C = eye(3);
D = rand(2, 7);
E = randn(6, 6);

% String
s1 = 'hello';
s2 = "world";

% Struct
st.x = 1;
st.y = zeros(3, 3);
st.z = 'name';

% Function handle
fh1 = @(x) x + 1;
fh2 = @(x, y) x * y;
fh3 = @sin;

% Cell
c1 = cell(3, 3);
c2 = {1, 'two', zeros(3,3); 4, 5, 6};

% Unknown (via unrecognized function)
u = unknown_func();

% ==========================================================================
% SECTION 2: Matrix arithmetic — valid operations
% ==========================================================================

% Scalar ops
r1 = a_scalar + b_scalar;
r2 = a_scalar - b_scalar;
r3 = a_scalar * b_scalar;
r4 = a_scalar / b_scalar;

% Matrix-scalar broadcasting
r5 = A + 1;
r6 = 2 * B;
r7 = C - 0.5;
r8 = D / 2;

% Element-wise ops (same-size)
F = ones(3, 3);
r9 = C + F;
r10 = C - F;
r11 = C .* F;
r12 = C ./ F;

% Matrix multiply (valid inner dim)
G = A * B;

% Transpose
H = A';
I = B';

% ==========================================================================
% SECTION 3: Deliberate dimension mismatches — trigger warnings
% ==========================================================================

% W_INNER_DIM_MISMATCH: A is 3x4, C is 3x3, inner dim 4 != 3
err_inner = A * C;

% W_ELEMENTWISE_MISMATCH: A is 3x4, C is 3x3
err_elem = A + C;

% W_ELEMENTWISE_MISMATCH with .* operator
err_elem2 = A .* C;

% ==========================================================================
% SECTION 4: All warning codes exercised
% ==========================================================================

% --- W_REASSIGN_INCOMPATIBLE ---
reass = zeros(3, 3);
reass = ones(4, 4);

% --- W_SUSPICIOUS_COMPARISON (matrix vs scalar) ---
if A == 5
    x_susp = 1;
end

% --- W_MATRIX_COMPARISON ---
if A == C
    x_matcmp = 1;
end

% --- W_LOGICAL_OP_NON_SCALAR ---
if A && B
    x_logic = 1;
end

% --- W_INDEXING_SCALAR ---
err_idx_scalar = a_scalar(1, 1);

% --- W_STRING_ARITHMETIC ---
err_str_arith = 'abc' + ones(3, 3);

% --- W_STRUCT_FIELD_NOT_FOUND ---
err_field = st.nonexistent;

% --- W_FIELD_ACCESS_NON_STRUCT ---
err_field2 = a_scalar.field;

% --- W_CURLY_INDEXING_NON_CELL ---
err_curly = A{1};

% --- W_CELL_ASSIGN_NON_CELL ---
A{1} = 5;

% --- W_UNKNOWN_FUNCTION ---
err_unknown = totally_unknown_fn(1, 2, 3);

% --- W_DIVISION_BY_ZERO ---
zero_val = 0;
err_div = 10 / zero_val;

% --- W_INDEX_OUT_OF_BOUNDS ---
err_oob = C(5, 1);

% --- W_POSSIBLY_NEGATIVE_DIM ---
neg_dim = -2;
err_neg = zeros(neg_dim, 3);

% --- W_ARITHMETIC_TYPE_MISMATCH (struct) ---
err_type1 = st + 5;

% --- W_ARITHMETIC_TYPE_MISMATCH (cell) ---
err_type2 = c1 * 3;

% --- W_ARITHMETIC_TYPE_MISMATCH (function_handle) ---
err_type3 = fh1 - 1;

% --- W_TRANSPOSE_TYPE_MISMATCH ---
err_trans = st';

% --- W_NEGATE_TYPE_MISMATCH ---
err_neg_type = -st;

% --- W_CONCAT_TYPE_MISMATCH ---
err_concat = [st; ones(3, 3)];

% --- W_RESHAPE_MISMATCH ---
err_reshape = reshape(zeros(3, 4), 5, 5);

% --- W_CONSTRAINT_CONFLICT ---
function result = constraint_conflict(A, B)
    C_inner = A * B;
    D_inner = A + B;
    result = C_inner;
end

% ==========================================================================
% SECTION 5: All builtins exercised
% ==========================================================================

% Constructors
b_zeros = zeros(5, 5);
b_ones = ones(3, 4);
b_eye = eye(4);
b_rand = rand(2, 3);
b_randn = randn(3, 3);
b_true = true(2, 2);
b_false = false(3, 1);
b_nan = nan(2, 2);
b_inf = inf(1, 5);
b_cell = cell(3, 2);
b_linspace = linspace(0, 1, 100);

% Trig functions (passthrough)
b_sin = sin(A);
b_cos = cos(A);
b_tan = tan(A);
b_asin = asin(A);
b_acos = acos(A);
b_atan = atan(A);
b_atan2 = atan2(A, A);

% Exponential/log (passthrough)
b_exp = exp(A);
b_log = log(A);
b_log2 = log2(A);
b_log10 = log10(A);

% Element-wise math (passthrough)
b_abs = abs(A);
b_sqrt = sqrt(A);
b_ceil = ceil(A);
b_floor = floor(A);
b_round = round(A);
b_sign = sign(A);
b_real = real(A);
b_imag = imag(A);

% Cumulative (passthrough)
b_cumsum = cumsum(A);
b_cumprod = cumprod(A);

% Reductions
b_sum = sum(A);
b_prod = prod(A);
b_mean = mean(A);
b_any = any(A);
b_all = all(A);
b_min = min(A);
b_max = max(A);

% Query functions
b_length = length(A);
b_numel = numel(A);
b_size = size(A);

% Type predicates (return scalar)
b_iscell = iscell(c1);
b_isscalar = isscalar(a_scalar);
b_isempty = isempty(A);
b_isnumeric = isnumeric(A);
b_islogical = islogical(A);
b_ischar = ischar(s1);
b_isnan = isnan(A);
b_isinf = isinf(A);
b_isfinite = isfinite(A);
b_issymmetric = issymmetric(C);

% Matrix manipulation
b_diag = diag(A);
b_inv = inv(C);
b_det = det(C);
b_norm = norm(A);
b_reshape = reshape(A, 4, 3);
b_repmat = repmat(A, 2, 3);
b_transpose = transpose(A);
b_diff = diff(A);

% Two-arg elementwise
b_mod = mod(A, 2);
b_rem = rem(A, 3);

% Dimension algebra builtins
b_kron = kron(C, eye(2));
b_blkdiag = blkdiag(C, eye(2));

% ==========================================================================
% SECTION 6: Symbolic dimensions
% ==========================================================================

M1 = zeros(n, m);
M2 = ones(m, k);
M3 = M1 * M2;
M4 = [M1; zeros(k, m)];
M5 = [M1, zeros(n, k)];

% Symbolic arithmetic
M6 = zeros(n + 1, m);
M7 = zeros(2 * n, m);

% ==========================================================================
% SECTION 7: Control flow — all constructs
% ==========================================================================

% --- If/else ---
if a_scalar > 0
    cf1 = zeros(3, 3);
else
    cf1 = ones(4, 4);
end

% --- If/elseif/else (IfChain) ---
if a_scalar > 10
    cf2 = zeros(2, 2);
elseif a_scalar > 5
    cf2 = ones(3, 3);
elseif a_scalar > 0
    cf2 = eye(4);
else
    cf2 = rand(5, 5);
end

% --- While loop ---
w = 0;
while w < 10
    w = w + 1;
end

% --- For loop (concrete bounds) ---
acc1 = zeros(1, 10);
for i = 1:10
    acc1(1, i) = i;
end

% --- For loop (symbolic bounds) ---
acc2 = zeros(1, n);
for j = 1:n
    acc2(1, j) = j;
end

% --- Switch/case ---
switch a_scalar
    case 1
        sw = zeros(2, 2);
    case 2
        sw = ones(3, 3);
    case 3
        sw = eye(4);
    otherwise
        sw = rand(5, 5);
end

% --- Try/catch ---
try
    tc1 = zeros(3, 3);
    tc2 = tc1 * ones(5, 5);
catch
    tc1 = eye(3);
end

% --- Nested loops with break/continue ---
for i = 1:5
    if i == 3
        continue;
    end
    for j = 1:5
        if j == 4
            break;
        end
        nested_val = i + j;
    end
end

% ==========================================================================
% SECTION 8: User-defined functions
% ==========================================================================

function [out1, out2] = multi_return(x, y)
    out1 = x + y;
    out2 = x * y;
end

function result = shape_maker(rows, cols)
    result = zeros(rows, cols);
end

function procedure_only(x)
    temp = x + 1;
end

% Call user-defined functions
[mr1, mr2] = multi_return(ones(3, 3), eye(3));
sm1 = shape_maker(3, 4);
sm2 = shape_maker(n, m);
procedure_only(A);

% Polymorphic caching: same function, different arg shapes
sm3 = shape_maker(5, 5);
sm4 = shape_maker(2, 7);

% ==========================================================================
% SECTION 9: Lambdas and closures
% ==========================================================================

% Basic lambda
lam1 = @(x) x + 1;
lr1 = lam1(5);
lr2 = lam1(ones(3, 3));

% Multi-param lambda
lam2 = @(x, y) x * y;
lr3 = lam2(ones(2, 3), ones(3, 4));

% Closure capture
base_val = zeros(3, 3);
lam3 = @(x) x + base_val;
lr4 = lam3(eye(3));

% Lambda stored in variable, called later
stored = @(n) zeros(n, n);
lr5 = stored(5);

% Function handle dispatch
handle_sin = @sin;
lr6 = handle_sin(ones(3, 3));

% Lambda in control flow (join)
if a_scalar > 0
    lam_cf = @(x) x + 1;
else
    lam_cf = @(x) x * 2;
end
lr7 = lam_cf(ones(3, 3));

% ==========================================================================
% SECTION 10: Cell arrays
% ==========================================================================

% Cell literal
cell_lit = {1, 'two', zeros(3,3); 4, 5, 6};

% Cell indexing
cell_elem = cell_lit{1, 3};

% Cell assignment
cell_lit{2, 1} = ones(4, 4);

% cell() constructor
empty_cell = cell(5, 5);

% Nested cells
nested_cell = {cell(2, 2), {1, 2, 3}};

% End keyword in cell indexing
cell_end = cell_lit{end};
cell_end2 = cell_lit{end - 1};

% ==========================================================================
% SECTION 11: Struct operations
% ==========================================================================

% Create struct
big_struct.name = 'test';
big_struct.data = zeros(10, 10);
big_struct.count = 42;
big_struct.flag = true(1, 1);

% Access fields
sn = big_struct.name;
sd = big_struct.data;
sc = big_struct.count;

% Reassign fields
big_struct.data = ones(5, 5);

% Nested struct
big_struct.nested.inner = zeros(2, 2);
inner_val = big_struct.nested.inner;

% ==========================================================================
% SECTION 12: Matrix literals (concatenation)
% ==========================================================================

% Horizontal concat
hc1 = [ones(3, 2), zeros(3, 4)];

% Vertical concat
vc1 = [ones(2, 5); zeros(3, 5)];

% Mixed concat
mc1 = [1, 2, 3; 4, 5, 6];
mc2 = [eye(2), zeros(2, 3); ones(1, 2), zeros(1, 3)];

% W_HORZCAT_ROW_MISMATCH
err_hc = [ones(2, 3), zeros(4, 3)];

% W_VERTCAT_COL_MISMATCH
err_vc = [ones(3, 2); zeros(3, 4)];

% String concat in matrix literal
str_cat = ['hello', ' ', 'world'];

% ==========================================================================
% SECTION 13: Indexing — all forms
% ==========================================================================

% Scalar indexing
idx1 = A(1, 2);

% Colon indexing
idx2 = A(:, 1);
idx3 = A(1, :);
idx4 = A(:, :);

% Range indexing
idx5 = A(1:2, :);
idx6 = A(:, 2:4);

% End keyword
idx7 = A(end, 1);
idx8 = A(1, end);
idx9 = A(1:end, 1);
idx10 = A(end-1, :);

% Symbolic range indexing
idx11 = M1(1:n, :);

% ==========================================================================
% SECTION 14: Interval analysis triggers
% ==========================================================================

% Division by zero
zero = 0;
err_div2 = 1 / zero;

% Out of bounds with known interval
arr = zeros(5, 5);
idx_var = 7;
err_oob2 = arr(idx_var, 1);

% Conditional refinement — should NOT warn
wide_var = 3;
if cond
    wide_var = 8;
end
if wide_var <= 5
    safe_idx = arr(wide_var, 1);
end

% For loop interval
for loop_i = 1:5
    loop_elem = arr(loop_i, 1);
end

% For loop interval OOB
small_arr = zeros(3, 3);
for loop_j = 1:5
    oob_elem = small_arr(loop_j, 1);
end

% Negative dimension
neg_n = -3;
err_neg2 = zeros(neg_n, 4);

% ==========================================================================
% SECTION 15: Constraint solving
% ==========================================================================

function constraint_demo(P, Q)
    R = P * Q;
    S = P + Q;
end

% ==========================================================================
% SECTION 16: Deep nesting — stability test
% ==========================================================================

if a_scalar > 0
    if a_scalar > 1
        if a_scalar > 2
            if a_scalar > 3
                if a_scalar > 4
                    deep = zeros(3, 3);
                else
                    deep = ones(3, 3);
                end
            else
                deep = eye(3);
            end
        else
            deep = rand(3, 3);
        end
    else
        deep = randn(3, 3);
    end
else
    deep = zeros(3, 3);
end

% --- Deeply nested loops ---
for d1 = 1:3
    for d2 = 1:3
        for d3 = 1:3
            for d4 = 1:3
                deep_loop = d1 + d2 + d3 + d4;
            end
        end
    end
end

% ==========================================================================
% SECTION 17: Edge cases
% ==========================================================================

% Empty matrix
empty_mat = [];

% Single element matrix
single = [42];

% Scalar from matrix index
scl_from_mat = A(1, 1);

% Transpose of scalar (identity)
t_scalar = a_scalar';

% Transpose of string
t_string = s1';

% Double transpose (identity for real)
dt = A'';

% Operations on unknown
u_plus = u + 1;
u_times = u * A;
u_trans = u';

% Chained operations
chain1 = (A' * A) + eye(4);
chain2 = sum(diag(C));

% ==========================================================================
% SECTION 18: Performance — large expressions
% ==========================================================================

% Many variables in sequence
v1 = zeros(3, 3);
v2 = v1 + ones(3, 3);
v3 = v2 * eye(3);
v4 = v3 - rand(3, 3);
v5 = v4 .* v1;
v6 = v5 ./ (v1 + 1);
v7 = v6';
v8 = [v7; v3];
v9 = [v8, [v1; v2]];
v10 = sum(v9);
v11 = reshape(v1, 9, 1);
v12 = repmat(v1, 3, 3);
v13 = kron(v1, eye(2));
v14 = blkdiag(v1, eye(2));
v15 = diag(v11);
v16 = inv(v1);
v17 = det(v1);
v18 = norm(v1);
v19 = abs(v1);
v20 = sqrt(v1);

% Large literal
big_lit = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
           11, 12, 13, 14, 15, 16, 17, 18, 19, 20;
           21, 22, 23, 24, 25, 26, 27, 28, 29, 30;
           31, 32, 33, 34, 35, 36, 37, 38, 39, 40;
           41, 42, 43, 44, 45, 46, 47, 48, 49, 50];

% Many function calls
fc1 = sin(cos(tan(exp(log(abs(A))))));
fc2 = max(min(sum(prod(A))));
fc3 = ceil(floor(round(sqrt(abs(A)))));

% ==========================================================================
% SECTION 19: Accumulation patterns (fixpoint)
% ==========================================================================

% Vertical accumulation
vacc = zeros(0, 3);
for ai = 1:5
    vacc = [vacc; ones(1, 3)];
end

% Horizontal accumulation
hacc = zeros(3, 0);
for bi = 1:5
    hacc = [hacc, ones(3, 1)];
end

% ==========================================================================
% SECTION 20: Mixed scenarios — real-world patterns
% ==========================================================================

% Matrix building pattern
function M = build_matrix(n)
    M = zeros(n, n);
    for i = 1:n
        for j = 1:n
            M(i, j) = i + j;
        end
    end
end

bm1 = build_matrix(5);
bm2 = build_matrix(n);

% Signal processing pattern
function y = filter_signal(x, h)
    y = zeros(1, length(x));
    for i = 1:length(x)
        y(1, i) = sum(x .* h);
    end
end

% Statistics pattern
function [mu, sigma] = compute_stats(data)
    mu = mean(data);
    sigma = sqrt(sum((data - mu) .* (data - mu)) / length(data));
end

% Matrix factorization pattern
function [L, U] = lu_simple(A)
    n_lu = size(A);
    L = eye(3);
    U = A;
end
