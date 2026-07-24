% Soundness guard: the shape domain cannot distinguish a logical mask from a
% numeric index vector, so a variable subscript must never fold to numel.
% B must stay matrix[None x 4], NOT matrix[500 x 4] -- if someone "optimises"
% the numeric-literal classifier into a syntactic blacklist (e.g. treating any
% comparison-derived variable as numeric), this is the test that catches it.
% EXPECT: B = matrix[None x 4]
% EXPECT: warnings = 1

A = randn(500, 4);
mask = A(:, 1) > 0;  % EXPECT_WARNING: W_SUSPICIOUS_COMPARISON
B = A(mask, :);
