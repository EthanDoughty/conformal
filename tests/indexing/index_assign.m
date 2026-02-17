% Basic indexed assignment: shape preservation
% EXPECT: warnings = 3

A = zeros(3, 4);
A(1, 1) = 5;
% EXPECT: A = matrix[3 x 4]

B = ones(5, 5);
B(2, 3) = 10;
B(:, 1) = zeros(5, 1);
B(1:3, :) = ones(3, 5);
% EXPECT: B = matrix[5 x 5]

% Single-arg indexing (linear)
C = zeros(2, 3);
C(4) = 99;
% EXPECT: C = matrix[2 x 3]

% Symbolic dimension preserved
D = zeros(n, m);
D(1, 1) = 0;
% EXPECT: D = matrix[n x m]

% Indexed assignment to scalar (type mismatch) — WARNING 1
x = 5;
x(1) = 10;

% Indexed assignment to struct (type mismatch) — WARNING 2
st.field = 1;
st(1, 1) = 5;

% Indexed assignment to unknown — WARNING 3 (W_UNKNOWN_FUNCTION)
u = unknown_func();
u(1, 1) = 5;
% EXPECT: u = unknown
