% Test: spline follows its query points (shares handleInterp1's rule),
% and roots(c) for a length-n coefficient vector returns (n-1)-by-1.
% EXPECT: warnings = 0

% --- spline: 3-arg form follows the query, like interp1 ---
x = zeros(1, 11);
y = zeros(1, 11);
xq = zeros(1, 41);

sq = spline(x, y, xq);
% EXPECT: sq = matrix[1 x 41]

% --- spline: 2-arg form returns a piecewise-polynomial struct, NOT
% args[1]'s shape. This is the one way the shared handler could go wrong. ---
pp = spline(x, y);
% EXPECT: pp = unknown

% --- roots: length-n coefficient row vector -> (n-1)-by-1 ---
r_row = roots([1 6 11 6]);
% EXPECT: r_row = matrix[3 x 1]

% --- roots: column coefficient vector gives the same answer ---
r_col = roots([1; 6; 11; 6]);
% EXPECT: r_col = matrix[3 x 1]

% --- roots: symbolic-length coefficient vector degrades cleanly ---
r_sym = roots(zeros(1, n));
% EXPECT: r_sym = matrix[(n-1) x 1]

% --- roots: a scalar coefficient (degree 0) has no roots ---
r_scalar = roots(5);
% EXPECT: r_scalar = matrix[0 x 1]
