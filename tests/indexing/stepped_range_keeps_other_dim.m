% The collapse fix is about the whole-shape gate, not about vectors: a
% stepped range with an unknown step folds to an unknown row extent, but
% the known column count must still survive.
% EXPECT: B = matrix[None x 5]
% EXPECT: warnings = 0

A = randn(8, 5);
k = randi(3);
B = A(1:k:8, :);
