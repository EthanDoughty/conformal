% Test: the n-ary size-preservation fix does not over-claim
% (the negative half: undecidable flags stay unknown, and the members
% that must NOT be in SIZE_PRESERVING_NARY_BUILTINS stay unknown too)
% EXPECT: warnings = 0

x = ones(1, 1000);
h = ones(1, 10);

% conv: a variable holding the flag is not statically decidable -> unknown
f = 'same';
y = conv(x, h, f);
% EXPECT: y = unknown

% conv: an unrecognized literal flag -> unknown
z = conv(x, h, 'bogus');
% EXPECT: z = unknown

% qr: a variable holding the flag on a TALL matrix is genuinely ambiguous,
% since full and economy differ when m > n.
g = 0;
A = zeros(10, 3);
[Qg, Rg] = qr(A, g);
% EXPECT: Qg = unknown
% EXPECT: Rg = unknown

% qr: same undecidable flag on a WIDE matrix -> full and economy coincide
% (m <= n), so the shape is exact even though the flag is not.
B = zeros(3, 10);
[Qw, Rw] = qr(B, g);
% EXPECT: Qw = matrix[3 x 3]
% EXPECT: Rw = matrix[3 x 10]

% atan2d broadcasts its two arguments; it must NOT be in the
% size-preserving set even though it sits in PASSTHROUGH_BUILTINS.
ad = atan2d(zeros(3, 1), zeros(1, 4));
% EXPECT: ad = unknown

% typecast resizes based on the target type; must NOT be size-preserving.
tc = typecast(zeros(1, 4), 'uint8');
% EXPECT: tc = unknown

% unique('rows') can shrink the row count; must NOT be size-preserving.
u = unique(zeros(4, 5), 'rows');
% EXPECT: u = unknown
