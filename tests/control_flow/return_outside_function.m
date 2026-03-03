% Test: return at script top level should warn
x = 1;
return; % EXPECT_WARNING: W_RETURN_OUTSIDE_FUNCTION
