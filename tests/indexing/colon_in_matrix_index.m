% Test: colon range expressions inside matrix literal index arguments
% Verifies that ':' is visible inside [...] when used as an index arg,
% which requires the colon_visible parameter threading in parse_index_arg.
%
% Key patterns:
%   A(1:3)        -- plain colon range as direct index arg (always worked)
%   A([1:3])      -- colon range inside matrix literal index arg (the fix)
%   A(1:3, [2:4]) -- mixed: direct range + matrix-literal range
%
% EXPECT: A = matrix[5 x 6]
% EXPECT: p = matrix[3 x 6]
% EXPECT: q = matrix[1 x None]
% EXPECT: r = matrix[3 x None]
% EXPECT: s = matrix[5 x 3]
% EXPECT: warnings = 0

A = zeros(5, 6);

% Direct colon range index (baseline -- always worked)
p = A(1:3, :);

% Colon inside a matrix literal used as index (the fix)
q = A([1:3]);

% Mixed: direct range row-index, matrix-literal col-index
r = A(1:3, [2:4]);

% Direct colon column index (baseline)
s = A(:, 1:3);
