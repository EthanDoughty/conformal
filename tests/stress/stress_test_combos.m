% Stress Test: Cross-Feature Combinations
% Exercises interactions between features that have not been tested together.
% EXPECT: warnings = 2

% ==========================================================================
% Lambda closure + struct field
% ==========================================================================
s.data = eye(3);
s.op = @(x) x + s.data;
combo1 = s.op(zeros(3, 3));
% GAP: Lambda called through struct field access loses lambda_id → returns unknown
% EXPECT: combo1 = unknown

% ==========================================================================
% Cell with function handles — dispatch through cell indexing
% ==========================================================================
handlers = {@sin, @cos};
% Cell element tracking retrieves function_handle for literal index
h1_result = handlers{1};
% EXPECT: h1_result = function_handle

% ==========================================================================
% Symbolic dims through function calls
% ==========================================================================
function M = make_sym(n)
    M = zeros(n, n);
end

sym_mat = make_sym(5);
% EXPECT: sym_mat = matrix[5 x 5]

% ==========================================================================
% Matrix literal with symbolic dims via function
% ==========================================================================
function R = sym_concat(n, m)
    top = zeros(n, m);
    bot = ones(n, m);
    R = [top; bot];
end

sc_result = sym_concat(3, 4);
% EXPECT: sc_result = matrix[6 x 4]

% ==========================================================================
% Lambda in for loop — each iteration defines a new closure
% ==========================================================================
total = zeros(3, 3);
for i = 1:5
    f = @(x) x + eye(3);
    total = total + f(zeros(3, 3));
end
% EXPECT: total = matrix[3 x 3]

% ==========================================================================
% Chained builtins: reshape(zeros(3,4), 4, 3)
% ==========================================================================
chain1 = reshape(zeros(3, 4), 4, 3);
% EXPECT: chain1 = matrix[4 x 3]

chain2 = diag(ones(1, 5));
% EXPECT: chain2 = matrix[5 x 5]

chain3 = inv(eye(4));
% EXPECT: chain3 = matrix[4 x 4]

% ==========================================================================
% Multi-return function + destructuring
% ==========================================================================
function [L, U] = lu_simple(A)
    L = eye(3);
    U = A;
end

[L_out, U_out] = lu_simple(zeros(3, 3));
product = L_out * U_out;
% EXPECT: L_out = matrix[3 x 3]
% EXPECT: U_out = matrix[3 x 3]
% EXPECT: product = matrix[3 x 3]

% ==========================================================================
% Linear (1D) indexing on 2D matrix
% ==========================================================================
M2d = zeros(3, 4);
lin_idx = M2d(5);
% EXPECT: lin_idx = scalar

% ==========================================================================
% Double transpose: A'' → same shape as A
% ==========================================================================
DT = zeros(3, 5);
dt_result = DT'';
% EXPECT: dt_result = matrix[3 x 5]

% ==========================================================================
% Transpose chain in matmul: (A' * A)
% ==========================================================================
TC = zeros(4, 3);
gram = TC' * TC;
% EXPECT: gram = matrix[3 x 3]

% ==========================================================================
% Operations on unknown — should all return unknown, no warnings
% ==========================================================================
u = unknown_func();
u_add = u + 1;
u_mul = u * zeros(3, 3);
u_neg = -u;
u_trans = u';
% EXPECT: u_add = unknown
% EXPECT: u_mul = unknown
% EXPECT: u_neg = unknown
% EXPECT: u_trans = unknown

% ==========================================================================
% String concat in matrix literal (horizontal char concat)
% ==========================================================================
str_cat = ['hello', ' ', 'world'];
% EXPECT: str_cat = string

% ==========================================================================
% Struct in cell — store and retrieve
% ==========================================================================
st.val = 42;
st.name = 'test';
cell_st = {st, 'hello', zeros(2, 2)};
retrieved = cell_st{1};
% Cell element tracking retrieves struct from literal index
% EXPECT: retrieved = struct{name: string, val: scalar}

% ==========================================================================
% Constraint conflict through function
% ==========================================================================
function z = conflict_func(A, B)
    z = A * B;
    w = A + B;
end

cf = conflict_func(zeros(3, 4), ones(4, 3));

% ==========================================================================
% Multiple chained function calls
% ==========================================================================
function out = double_it(x)
    out = x + x;
end

function out2 = quad_it(x)
    out2 = double_it(double_it(x));
end

q = quad_it(zeros(2, 3));
% EXPECT: q = matrix[2 x 3]

% ==========================================================================
% Transpose interaction with elementwise ops
% ==========================================================================
P = zeros(3, 4);
Q = ones(4, 3);
ew = P' .* Q;
% EXPECT: ew = matrix[4 x 3]

% ==========================================================================
% Concatenation with transposed matrices
% ==========================================================================
R1 = zeros(3, 2);
R2 = ones(3, 2);
hcat = [R1, R2];
% EXPECT: hcat = matrix[3 x 4]

vcat = [R1'; R2'];
% EXPECT: vcat = matrix[4 x 3]
