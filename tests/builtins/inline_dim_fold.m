% Test: numel/length/size written INLINE in a dimension argument fold to a
% concrete extent, just like the hoisted form n = numel(v); zeros(1, n)
% already does. Exercises every consumer of exprToDimIrCtx at once.
% EXPECT: warnings = 0
% EXPECT: v = matrix[1 x 7]
% EXPECT: A = matrix[3 x 4]
% EXPECT: z1 = matrix[1 x 7]
% EXPECT: z2 = matrix[1 x 7]
% EXPECT: z3 = matrix[7 x 1]
% EXPECT: z4 = matrix[7 x 7]
% EXPECT: z5 = matrix[1 x 7]
% EXPECT: z6 = matrix[1 x 7]
% EXPECT: z7 = matrix[7 x 1]
% EXPECT: z8 = matrix[1 x 49]
% EXPECT: z9 = matrix[3 x 4]
% EXPECT: z10 = matrix[1 x 12]

v = zeros(1, 7);
A = zeros(3, 4);

z1 = zeros(1, numel(v));
z2 = zeros(1, length(v));
z3 = zeros(size(v, 2), 1);
z4 = eye(numel(v));
z5 = A(1:numel(v));            % linear-index range: EvalExpr's index-extent path
z6 = linspace(0, 1, numel(v));
z7 = reshape(v, numel(v), 1);  % was unknown: handleReshape returned None on an Unknown dim
z8 = repmat(v, 1, numel(v));
z9 = zeros(size(A, 1), size(A, 2));
z10 = zeros(1, numel(A));
