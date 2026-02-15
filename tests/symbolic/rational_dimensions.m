% Test: Rational symbolic dimensions (internal, requires manual verification)
% Note: This test demonstrates rational dimension tracking but cannot be
% automatically verified because expr_to_dim_ir doesn't support division yet.
% The Python unit test (tests/test_shapes.py) validates the SymDim internals.
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
% EXPECT: B = matrix[5 x 5]

% Placeholder test — rational dims work internally but not exposed via division operator
A = zeros(3, 3);
B = zeros(5, 5);
