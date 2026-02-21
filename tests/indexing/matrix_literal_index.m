% Test: matrix literal used as index argument
% Key fix: colon visible inside matrix literal in index arg (colon_visible parameter threading)
% A([1:3]) parses correctly; plain colon ranges also work; 0 warnings throughout
% EXPECT: A = matrix[10 x 5]
% EXPECT: B = matrix[1 x 3]
% EXPECT: C = matrix[None x None]
% EXPECT: D = matrix[3 x 5]
% EXPECT: E = matrix[10 x 3]
% EXPECT: F = scalar
% EXPECT: G = matrix[1 x None]
% EXPECT: warnings = 0

A = rand(10, 5);

% Matrix literal as 1-arg linear index: result has same shape as the index vector
B = A([1 2 3]);

% Matrix literal as 2-arg subscript index: result shape is matrix[None x None]
C = A([1 2 3], [4 5]);

% Plain colon ranges inside index args -- these work and produce no warnings
D = A(1:3, :);
E = A(:, 2:4);
F = A(1:3);

% Matrix literal with colon range inside: A([1:4]) -- colon visible inside [...]
G = A([1:4]);
