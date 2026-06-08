% Test: Sound length inference for fractional-literal stepped ranges (Gap A).
% Ports Cleve Moler's colonop algorithm for bit-identical results with MATLAB.
%
% EXPECT: warnings = 1
% EXPECT: d = matrix[1 x 391]
% EXPECT: e = matrix[1 x 361]
% EXPECT: g = matrix[1 x 5]
% EXPECT: h = matrix[1 x 3]
% EXPECT: fp3 = matrix[1 x 4]
% EXPECT: mid = matrix[1 x 3]
% EXPECT: z = matrix[1 x None]

% Motivating case: 0.025:0.0025:1
% (1-0.025)/0.0025 = 390.0 exactly — round=390, len=391
d = 0.025:0.0025:1;

% Slightly different start: 0.1:0.0025:1
% (1-0.1)/0.0025 = 360.0 exactly — round=360, len=361
e = 0.1:0.0025:1;

% The win: d is 1x391, e is 1x361 — mismatch must warn
f = d + e;  % EXPECT_WARNING: W_ELEMENTWISE_MISMATCH

% Clean fractional step: 0:0.5:2 -> 0, 0.5, 1.0, 1.5, 2.0 -> 5 elements
g = 0:0.5:2;

% Negative fractional step: 1:-0.5:0 -> 1, 0.5, 0 -> 3 elements
h = 1:-0.5:0;

% FP boundary battery -------------------------------------------------------

% 0:0.1:0.3: MATLAB returns 4 (not 3 via naive floor).
% (0.3-0)/0.1 = 2.9999999999999996, round = 3.
% Endpoint check: 0+3*0.1 = 0.30000000000000004 vs 0.3;
% sgn*(0.30000000000000004-0.3) = 3.7e-17, tol = 2*eps*0.3 = 1.33e-16.
% 3.7e-17 <= 1.33e-16, so first guard does NOT fire.
% sgn*(0+4*0.1-0.3) = 0.10000...>0, second guard does not fire.  n=3, len=4.
fp3 = 0:0.1:0.3;

% Clean fractional with an inexact quotient: 0:0.2:0.5.
% (0.5-0)/0.2 = 2.4999999999999996 in IEEE-754 (not 2.5), rounds to 2.
% Endpoint guard: 0+3*0.2 = 0.6 exceeds 0.5, so n stays 2, len=3.
% Elements: 0, 0.2, 0.4 (0.6 exceeds the endpoint).
mid = 0:0.2:0.5;

% Gap B boundary: variable-fractional step stays Unknown (no warning).
% Lambda is a float variable; the value domain is integer-only, so the
% analyzer cannot fold it and must leave z as 1 x None.
Lambda = 0.2;
z = 0:Lambda/400:Lambda/4;
