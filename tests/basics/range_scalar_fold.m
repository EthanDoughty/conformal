% Test: Integer-singleton variable folding into stepped ranges (Gap B, Slice 1).
% Teaches tryExtractConstFloatCtx to resolve Var from valueRanges singletons,
% so e.g. n=10; 0:n/4:n resolves to 0:2.5:10 -> 1x5.
%
% EXPECT: warnings = 1
% EXPECT: r = matrix[1 x 5]
% EXPECT: a = matrix[1 x 5]
% EXPECT: b = matrix[1 x 4]
% EXPECT: branch_r = matrix[1 x None]
% EXPECT: float_z = matrix[1 x None]

% Happy path: n is an integer singleton -> n/4 = 2.5 -> 0:2.5:10 -> 1x5.
n = 10;
r = 0:n/4:n;

% Mismatch via folded scalar: a is 1x5, b is 1x4 -> W_ELEMENTWISE_MISMATCH.
a = 0:n/4:n;
b = zeros(1,4);
c = a + b;  % EXPECT_WARNING: W_ELEMENTWISE_MISMATCH

% SOUNDNESS: branch-dependent integer -- after the if/else join, n is a
% non-singleton interval [10,20], so 0:n/4:n must NOT fold -> 1 x None.
% No warning emitted.
if cond
    m = 10;
else
    m = 20;
end
branch_r = 0:m/4:m;

% No-regression (float constant): Lambda=0.2 is not stored in valueRanges
% (integer-only domain in Slice 1), so float_z must stay 1 x None.
Lambda = 0.2;
float_z = 0:Lambda/400:Lambda/4;
