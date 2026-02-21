% Symbolic mismatch: inner dim is n, row dim of B is n+1 (always conflicts).
% Use script-level symbolic variables via zeros(n, ...) not possible directly,
% so rely on the function form with concrete call to trigger analysis.
function y = test_sym(n)
    A = zeros(n, n);
    B = zeros(n+1, n);
    y = A * B;
end

test_sym(5);
% EXPECT: warnings = 1
