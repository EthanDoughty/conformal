% EXPECT: warnings = 1
A = zeros(5, 5);
for i = 1:10
    x = A(i, 1);  % EXPECT_WARNING: W_INDEX_OUT_OF_BOUNDS
end
