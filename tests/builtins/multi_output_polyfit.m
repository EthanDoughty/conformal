% Test: Three-output polyfit and two-output polyval (Cluster B2).
% [p,S,mu] = polyfit(x,y,n): p is 1-by-(n+1), same as the 1-output form.
% S is the open error-estimate struct (R, df, normr, rsquared); mu is the
% 2-by-1 centering/scaling vector [mean(x); std(x)].
% [yfit,delta] = polyval(p,xq,S,mu): delta matches yfit's shape.

% EXPECT: warnings = 0

x = linspace(0, 10, 50);
y = linspace(0, 10, 50);
[p, S, mu] = polyfit(x, y, 5);
% EXPECT: p = matrix[1 x 6]
% EXPECT: mu = matrix[2 x 1]
% EXPECT: S = struct{R: matrix[6 x 6], df: scalar, normr: scalar, rsquared: scalar, ...}

sn = S.normr;
% EXPECT: sn = scalar

xq = zeros(200, 1);
[yfit, delta] = polyval(p, xq, S, mu);
% EXPECT: yfit = matrix[200 x 1]
% EXPECT: delta = matrix[200 x 1]
