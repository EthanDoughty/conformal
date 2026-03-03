% Test: break outside loop should warn
x = 1;
break; % EXPECT_WARNING: W_BREAK_OUTSIDE_LOOP
