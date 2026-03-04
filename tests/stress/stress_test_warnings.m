% Stress Test: Warning Code Coverage
% Exercises thin-coverage and rarely-triggered warning codes.
% Each section targets a specific W_* code.
% EXPECT: warnings = 29

% ==========================================================================
% W_TOO_MANY_INDICES: 3+ indices on a 2D matrix
% ==========================================================================
A = zeros(3, 4);
too_many = A(1, 2, 3);  % EXPECT_WARNING: W_TOO_MANY_INDICES

% ==========================================================================
% W_INVALID_RANGE: end < start in range
% ==========================================================================
B = ones(5, 5);
inv_range = B(5:2, :);  % EXPECT_WARNING: W_INVALID_RANGE

% ==========================================================================
% W_NON_SCALAR_INDEX: matrix used as index argument
% ==========================================================================
C = zeros(4, 4);
idx_mat = ones(2, 2);
non_scalar = C(idx_mat, 1);  % EXPECT_WARNING: W_NON_SCALAR_INDEX

% ==========================================================================
% W_RANGE_NON_SCALAR: non-scalar range endpoint
% ==========================================================================
D = zeros(5, 5);
lo = ones(2, 2);
hi = ones(3, 3);
range_nonscalar = D(lo:hi, :);  % EXPECT_WARNING: W_RANGE_NON_SCALAR

% ==========================================================================
% W_MULTI_ASSIGN_BUILTIN: destructuring from builtin
% ==========================================================================
[ma1, ma2] = sin(5);  % EXPECT_WARNING: W_MULTI_ASSIGN_COUNT_MISMATCH

% ==========================================================================
% W_MULTI_ASSIGN_NON_CALL: destructuring from non-call expr
% ==========================================================================
[nc1, nc2] = 5;  % EXPECT_WARNING: W_MULTI_ASSIGN_NON_CALL

% ==========================================================================
% W_PROCEDURE_IN_EXPR: procedure used in expression
% ==========================================================================
function do_nothing(x)
    y_local = x + 1;
end

proc_result = do_nothing(5);  % EXPECT_WARNING: W_PROCEDURE_IN_EXPR

% ==========================================================================
% W_LAMBDA_ARG_COUNT_MISMATCH: wrong arg count
% ==========================================================================
f_one = @(x) x + 1;
lam_bad = f_one(1, 2, 3);  % EXPECT_WARNING: W_LAMBDA_ARG_COUNT_MISMATCH

% ==========================================================================
% W_RECURSIVE_LAMBDA: self-referencing lambda (must call to trigger)
% ==========================================================================
rec_lam = @(x) rec_lam(x - 1);  % EXPECT_WARNING: W_RECURSIVE_LAMBDA
rec_result = rec_lam(5);

% ==========================================================================
% W_CONCAT_TYPE_MISMATCH: function handle in matrix literal
% ==========================================================================
fh = @(x) x;
concat_fh = [fh; ones(3, 3)];  % EXPECT_WARNING: W_CONCAT_TYPE_MISMATCH

% ==========================================================================
% W_CONCAT_TYPE_MISMATCH: cell in matrix literal
% ==========================================================================
c = {1, 2, 3};
concat_cell = [c; ones(3, 3)];  % EXPECT_WARNING: W_CONCAT_TYPE_MISMATCH

% ==========================================================================
% W_ARITHMETIC_TYPE_MISMATCH: string * matrix (triggers W_STRING_ARITHMETIC)
% ==========================================================================
s1 = 'hello';
type_str_mul = s1 * A;  % EXPECT_WARNING: W_STRING_ARITHMETIC

% ==========================================================================
% W_ARITHMETIC_TYPE_MISMATCH: cell / scalar
% ==========================================================================
c2 = {1, 2};
type_cell_div = c2 / 5;  % EXPECT_WARNING: W_ARITHMETIC_TYPE_MISMATCH

% ==========================================================================
% W_TRANSPOSE_TYPE_MISMATCH: cell transpose
% ==========================================================================
c3 = {1, 2, 3};
type_cell_t = c3';  % EXPECT_WARNING: W_TRANSPOSE_TYPE_MISMATCH

% ==========================================================================
% W_NEGATE_TYPE_MISMATCH: -cell
% ==========================================================================
c4 = cell(2, 2);
neg_cell = -c4;  % EXPECT_WARNING: W_NEGATE_TYPE_MISMATCH

% ==========================================================================
% W_NEGATE_TYPE_MISMATCH: -function_handle
% ==========================================================================
fh2 = @sin;
neg_fh = -fh2;  % EXPECT_WARNING: W_NEGATE_TYPE_MISMATCH

% ==========================================================================
% return in switch inside script (valid MATLAB, no warning)
% ==========================================================================
x_switch = 2;
switch x_switch
    case 1
        return;
    case 2
        y_sw = 10;
end

% ==========================================================================
% W_LOGICAL_OP_NON_SCALAR: matrix && matrix
% ==========================================================================
E = ones(3, 3);
F = zeros(3, 3);
log_bad = E && F;  % EXPECT_WARNING: W_LOGICAL_OP_NON_SCALAR

% ==========================================================================
% W_SUSPICIOUS_COMPARISON: matrix vs scalar
% ==========================================================================
G = ones(4, 4);
susp1 = G > 0;  % EXPECT_WARNING: W_SUSPICIOUS_COMPARISON
susp2 = G == 5;  % EXPECT_WARNING: W_SUSPICIOUS_COMPARISON

% ==========================================================================
% W_REASSIGN_INCOMPATIBLE: scalar → matrix triggers warning
% GAP: matrix → string does NOT trigger W_REASSIGN_INCOMPATIBLE
% ==========================================================================
ra = zeros(3, 3);
ra = 'hello';

ra2 = 5;
ra2 = zeros(4, 4);

% ==========================================================================
% W_INDEXING_SCALAR: indexing applied to scalar
% ==========================================================================
sc = 5;
sc_idx = sc(1, 2);

% ==========================================================================
% W_STRUCT_FIELD_NOT_FOUND: missing field
% ==========================================================================
st.x = 1;
st.y = 2;
missing = st.z;  % EXPECT_WARNING: W_STRUCT_FIELD_NOT_FOUND

% ==========================================================================
% W_FIELD_ACCESS_NON_STRUCT: field access on scalar
% ==========================================================================
ns = 42;
ns_field = ns.foo;  % EXPECT_WARNING: W_FIELD_ACCESS_NON_STRUCT

% ==========================================================================
% W_CURLY_INDEXING_NON_CELL: curly on non-cell
% ==========================================================================
not_cell = zeros(3, 3);
curly_bad = not_cell{1};  % EXPECT_WARNING: W_CURLY_INDEXING_NON_CELL

% ==========================================================================
% W_CELL_ASSIGN_NON_CELL: cell assign to non-cell
% ==========================================================================
not_cell2 = 5;
not_cell2{1} = 10;  % EXPECT_WARNING: W_CELL_ASSIGN_NON_CELL

% ==========================================================================
% W_UNKNOWN_FUNCTION: unrecognized function name
% ==========================================================================
unk = gobbledygook(1, 2, 3);  % EXPECT_WARNING: W_UNKNOWN_FUNCTION

% ==========================================================================
% W_FUNCTION_ARG_COUNT_MISMATCH: wrong arg count to user function
% ==========================================================================
function result = takes_two(a, b)
    result = a + b;
end

bad_args = takes_two(1, 2, 3);  % EXPECT_WARNING: W_FUNCTION_ARG_COUNT_MISMATCH

% ==========================================================================
% W_STRING_ARITHMETIC: string + matrix
% ==========================================================================
s2 = "world";
str_add = s2 + ones(3, 3);  % EXPECT_WARNING: W_STRING_ARITHMETIC

% ==========================================================================
% W_MATRIX_COMPARISON: matrix vs matrix
% ==========================================================================
H = ones(3, 3);
I_mat = zeros(3, 3);
mat_cmp = H > I_mat;  % EXPECT_WARNING: W_MATRIX_COMPARISON

% ==========================================================================
% W_BREAK_OUTSIDE_LOOP / W_CONTINUE_OUTSIDE_LOOP
% NOTE: break/continue at program level are invalid MATLAB. Placed last
% because FlowBreak/FlowContinue halt subsequent analysis.
% ==========================================================================
break; % EXPECT_WARNING: W_BREAK_OUTSIDE_LOOP
continue; % EXPECT_WARNING: W_CONTINUE_OUTSIDE_LOOP
