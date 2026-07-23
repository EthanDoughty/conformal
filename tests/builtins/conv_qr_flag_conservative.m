% Test: conv/qr flag handling stays conservative on undecidable flags
% (the fix must not over-claim: only statically-literal flags are honored)
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
