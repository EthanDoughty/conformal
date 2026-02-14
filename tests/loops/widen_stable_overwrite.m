% Test: Loop body overwrites variable with stable but different shape
% Body always produces matrix[4 x 5] regardless of A's value,
% but pre-loop A is matrix[2 x 3], so post-loop join widens both dims.
% EXPECT: warnings = 1
% EXPECT: A = matrix[4 x 5]
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = matrix[None x None]

A = zeros(2, 3);
for i = 1:n
    A = ones(4, 5);
end
