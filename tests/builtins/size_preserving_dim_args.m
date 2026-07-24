% Test: SIZE_PRESERVING_NARY_BUILTINS -- n-ary dim/option forms of sort,
% cumsum, cumprod, circshift preserve args[0]'s size regardless of the
% trailing dimension or option arguments.
% EXPECT: warnings = 1

A = ones(4, 5);

cs_dim1 = cumsum(A, 1);
% EXPECT: cs_dim1 = matrix[4 x 5]
cs_dim2 = cumsum(A, 2);
% EXPECT: cs_dim2 = matrix[4 x 5]

srt_dim1 = sort(A, 1);
% EXPECT: srt_dim1 = matrix[4 x 5]
srt_dim2 = sort(A, 2);
% EXPECT: srt_dim2 = matrix[4 x 5]
srt_desc = sort(A, 'descend');
% EXPECT: srt_desc = matrix[4 x 5]
srt_dim_desc = sort(A, 2, 'descend');
% EXPECT: srt_dim_desc = matrix[4 x 5]

cp_dim2 = cumprod(A, 2);
% EXPECT: cp_dim2 = matrix[4 x 5]

cshift1 = circshift(A, 1);
% EXPECT: cshift1 = matrix[4 x 5]
cshift2 = circshift(A, 2, 2);
% EXPECT: cshift2 = matrix[4 x 5]

% Vector cases: the dim/option argument never changes the extent.
srt_vec = sort(ones(1, 7), 2);
% EXPECT: srt_vec = matrix[1 x 7]
cshift_vec = circshift(zeros(8, 1), 3);
% EXPECT: cshift_vec = matrix[8 x 1]

% Symbolic case: still just args[0]'s shape.
cs_sym = cumsum(zeros(n, m), 1);
% EXPECT: cs_sym = matrix[n x m]

% The n-ary passthrough dispatch now walks the tail arguments too, so a
% dimension error nested inside args[0] is no longer silently dropped.
bad = circshift(zeros(2, 3) * zeros(4, 5), 1);  % EXPECT_WARNING: W_INNER_DIM_MISMATCH
