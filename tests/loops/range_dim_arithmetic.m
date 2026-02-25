% Test: Arithmetic on range dimensions
% B starts at 2x3, conditional loop produces range rows; C = [B; zeros(2,3)] adds 2
% Fixpoint: B = matrix[2.. x 3], C = matrix[4.. x 3]
% Default: single pass, if-join produces Unknown for rows
% EXPECT: warnings = 1
% EXPECT: B = matrix[None x 3]
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: B = matrix[2.. x 3]
% EXPECT_FIXPOINT: C = matrix[4.. x 3]

B = zeros(2, 3);
for i = 1:10
    if flag
        B = [B; zeros(1, 3)];
    end
end
C = [B; zeros(2, 3)];
