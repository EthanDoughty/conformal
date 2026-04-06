% Test: backward stencil A(i-1) inside for i=2:n should produce no warnings.
% Upper: i <= n + 0, index i-1 has exprOffset=-1, totalOffset=0+(-1)=-1<=0. SUPPRESSED.
% Lower: i >= 2 + 0, index i-1 has exprOffset=-1, lo + totalOffset = 2+(-1) = 1 >= 1. SUPPRESSED.
% EXPECT: warnings = 0
function test(A)
    n = size(A, 1);
    for i = 2:n
        x = A(i-1, 1);
    end
end
