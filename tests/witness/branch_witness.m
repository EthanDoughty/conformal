function y = test_branch(n)
    A = zeros(n, n);
    if n > 3
        B = zeros(n+1, n);
        C = A * B;  % EXPECT_WARNING: W_INNER_DIM_MISMATCH
    end
    y = 0;
end

test_branch(5);
% EXPECT: warnings = 1
