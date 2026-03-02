% Function handle variable not resolved (opaque handle fires W_LAMBDA_CALL_APPROXIMATE)
% EXPECT: warnings >= 1
f = @unknownFunc;
result = f(5);  % EXPECT_WARNING: W_LAMBDA_CALL_APPROXIMATE
