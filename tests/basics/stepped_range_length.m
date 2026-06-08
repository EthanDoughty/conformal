% Test: Concrete length inference for constant integer stepped ranges.
% a:step:b in expression context should produce a concrete column count when
% step is an integer literal and start/stop are concrete integers.
%
% EXPECT: warnings = 1
% EXPECT: a = matrix[1 x 6]
% EXPECT: b = matrix[1 x 4]
% EXPECT: c = unknown
% EXPECT: e = matrix[1 x 5]
% EXPECT: f = matrix[1 x 0]

% Positive step: floor((10-0)/2)+1 = 6
a = 0:2:10;

% Negative step: floor((10-1)/3)+1 = 4
b = 10:-3:1;

% Mismatch: a is 1x6, b is 1x4 — elementwise add must warn
c = a + b;  % EXPECT_WARNING: W_ELEMENTWISE_MISMATCH

% Guard: decreasing by 1 — floor((5-1)/1)+1 = 5
e = 5:-1:1;

% Empty range: start > stop with positive step — length 0
f = 1:1:0;
