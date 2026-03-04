% Test: W_SUSPICIOUS_COMPARISON suppressed when matrix is fully unknown.
% MODE: strict
% EXPECT: warnings = 0

function test_comparison_unknown()
    Z = [];
    M = 5;
    for t = 1:M
        Z = [Z; ones(t, 1)];
    end
    % Z is matrix[None x None] after fixpoint widening
    % Comparing Z == scalar should NOT fire W_SUSPICIOUS_COMPARISON
    comp_size = length(find(Z == 3));
end
