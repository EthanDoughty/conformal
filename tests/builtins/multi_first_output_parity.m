% Test: First-output fallback contract for unmodeled multi-output builtins.
% gradient has no multi-output handler, so evalMultiBuiltinCall declines.
% analyzeAssignMulti's fallback (StmtFuncAnalysis.fs) must give the first
% target the shape the single-output form produces rather than erasing it
% to unknown, while the remaining targets stay unknown and the arity
% warning still fires (Decision 2, option A). This is the guarantee that
% stops the "second output appears, first output degrades to unknown" bug
% class from recurring for every future unmodeled multi-output builtin.

% EXPECT: warnings = 1
% EXPECT: g1 = matrix[4 x 6]
% EXPECT: g2 = unknown

F = zeros(4, 6);
[g1, g2] = gradient(F);  % EXPECT_WARNING: W_MULTI_ASSIGN_COUNT_MISMATCH
