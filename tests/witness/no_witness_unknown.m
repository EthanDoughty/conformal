% Unknown dims — dims_definitely_conflict returns False, so no warning, no witness.
function test_unknown(A, B)
    C = A * B;
end
% EXPECT: warnings = 0
