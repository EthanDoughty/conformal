% Regression: clear/clearvars must also evict valueRanges
% and DimEquiv concrete so that the dim-conflict checker cannot resolve stale values.
% EXPECT: warnings = 0

% Variant A: clear <name> (plain range 1:n after clearing n).
n15a = 10;
clear n15a
r15a = 1:n15a;
s15a = zeros(1, 7);
chk15a = r15a + s15a;
% EXPECT: r15a = matrix[1 x n15a]

% Variant B: clearvars (clears everything, n becomes undefined).
n15b = 10;
clearvars
r15b = 1:n15b;
s15b = zeros(1, 7);
chk15b = r15b + s15b;
% EXPECT: r15b = matrix[1 x n15b]
