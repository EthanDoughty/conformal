% Test: stencil access A(i+1) inside for i=1:n-1 should produce no warnings.
% The pentagon records i <= n + (-1), and the index i+1 has exprOffset=+1,
% so totalOffset = -1+1 = 0 <= 0, which proves i+1 <= n. SUPPRESSED.
% EXPECT: warnings = 0
function test(A)
    n = size(A, 1);
    for i = 1:n-1
        x = A(i+1, 1);
    end
end
