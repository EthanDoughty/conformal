% Test: Two-output xcorr and its length/orientation rules (Cluster B2).
% Correlation output length is 2*maxlag+1, maxlag defaulting to
% max(numel(x), numel(y)) - 1. The correlation follows the input's
% orientation (row in, row out; column in, column out); lags is ALWAYS
% a row vector. A scalar second argument is maxlag, not a second signal.

% EXPECT: warnings = 0

y = zeros(1, 512);
x = zeros(1, 512);
[r, lags] = xcorr(y, x);
% EXPECT: r = matrix[1 x 1023]
% EXPECT: lags = matrix[1 x 1023]

% Column inputs: correlation follows orientation, lags stays a row.
yc = zeros(512, 1);
xc = zeros(512, 1);
[rc, lagsc] = xcorr(yc, xc);
% EXPECT: rc = matrix[1023 x 1]
% EXPECT: lagsc = matrix[1 x 1023]

% Unequal lengths: outLen uses the max of the two.
y2 = zeros(1, 400);
[r2, lags2] = xcorr(y2, x);
% EXPECT: r2 = matrix[1 x 1023]
% EXPECT: lags2 = matrix[1 x 1023]

% Scalar second argument is maxlag (MATLAB's own disambiguation rule).
x100 = zeros(1, 100);
[r3, lags3] = xcorr(x100, 25);
% EXPECT: r3 = matrix[1 x 51]
% EXPECT: lags3 = matrix[1 x 51]

% Single-output form on a column input.
x100c = zeros(100, 1);
r4 = xcorr(x100c);
% EXPECT: r4 = matrix[199 x 1]
