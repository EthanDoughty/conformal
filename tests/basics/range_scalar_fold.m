% Test: Constant folding into stepped ranges (Gap B, Slice 2 conservative redesign).
% tryExtractConstFloatCtx resolves Var from trustedConsts (float store),
% so e.g. n=10; 0:n/4:n -> 1x5 and Lambda=0.2; 0:Lambda/400:Lambda/4 -> 1x101.
%
% EXPECT: warnings = 1
% EXPECT: r = matrix[1 x 5]
% EXPECT: a = matrix[1 x 5]
% EXPECT: b = matrix[1 x 4]
% EXPECT: branch_r = matrix[1 x None]
% EXPECT: float_z = matrix[1 x 101]

% Happy path: n is an integer constant -> n/4 = 2.5 -> 0:2.5:10 -> 1x5.
n = 10;
r = 0:n/4:n;

% Mismatch via folded scalar: a is 1x5, b is 1x4 -> W_ELEMENTWISE_MISMATCH.
a = 0:n/4:n;
b = zeros(1,4);
c = a + b;  % EXPECT_WARNING: W_ELEMENTWISE_MISMATCH

% SOUNDNESS: branch-dependent -- after the if/else join, m is removed from
% trustedConsts (assignments happened inside branches), so 0:m/4:m must NOT fold.
% No warning emitted.
if cond
    m = 10;
else
    m = 20;
end
branch_r = 0:m/4:m;

% Float constant: Lambda=0.2 is stored in trustedConsts at depth 0.
% 0:Lambda/400:Lambda/4 = 0:0.0005:0.05 -> 1x101.
Lambda = 0.2;
float_z = 0:Lambda/400:Lambda/4;
