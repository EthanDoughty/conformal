% Test: continue outside loop should warn
x = 1;
continue; % EXPECT_WARNING: W_CONTINUE_OUTSIDE_LOOP
