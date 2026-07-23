% Test: Argument diagnostics through the multi-assign fallback path, and
% the decline-before-evaluate contract on handlers that reject an out-of-
% range output count (lu here; find and unique follow the same pattern).
%
% First: a real dimension error inside the arguments of an unmodeled
% multi-output call must be REPORTED. max has no 3-output handler, so
% today only the arity warning fires and the inner-dim mismatch inside
% the argument is silently swallowed. The fallback now evaluates the
% argument via the single-output handler, so both warnings must fire.
%
% Second: lu's handler used to evaluate its argument BEFORE checking
% arity, so a 4-output lu call (invalid: lu supports 2 or 3) would
% evaluate the argument as a side effect of declining. The arity guard is
% now hoisted above the evaluation, so it declines cleanly without
% touching the argument. lu has no single-output shape rule either (it is
% only ever reached through its multi handler), so the fallback cannot
% re-derive the argument's shape: the inner-dim mismatch that used to
% leak out via the old evaluate-before-decline bug is gone for this line,
% and nothing here fires it twice. Reverting the lu reorder would bring
% that inner-dim warning back on this line, which is what this half pins.

% EXPECT: warnings = 3

[a, b, c] = max(ones(2, 3) * ones(4, 5));  % EXPECT_WARNING: W_INNER_DIM_MISMATCH  % EXPECT_WARNING: W_MULTI_ASSIGN_COUNT_MISMATCH

[d, e, f, g] = lu(ones(2, 3) * ones(4, 5));  % EXPECT_WARNING: W_MULTI_ASSIGN_COUNT_MISMATCH  % EXPECT_NO_WARNING: W_INNER_DIM_MISMATCH
