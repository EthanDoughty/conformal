% Regression: plain range a:b length computation.
%
% Bug 8: fractional endpoint must use floor(b-a)+1, not round(b)-round(a)+1.
%   1:5.9 -> floor(5.9-1)+1 = 5, i.e. [1 2 3 4 5].
%   0:2.5 -> floor(2.5-0)+1 = 3, i.e. [0 1 2].
%
% Bug 9: endpoint beyond Int64 range (e.g. 0:1e19) must yield 1xNone,
%   not a wrapped negative length.
%
% EXPECT: warnings = 0

% Bug 8a: 1:5.9 -> 1x5 (not 1x6).
a8 = 1;
b8 = 5.9;
z8 = a8:b8;
q8 = ones(1, 5);
chk8 = [z8; q8];
% EXPECT: z8 = matrix[1 x 5]
% EXPECT: chk8 = matrix[2 x 5]

% Bug 8b: 0:2.5 -> 1x3.
z8b = 0:2.5;
% EXPECT: z8b = matrix[1 x 3]

% Bug 9: 0:1e19 -> 1xNone (no negative/wrapped length).
z9 = [0:1e19; ones(1,3)];
% EXPECT: z9 = matrix[2 x None]
