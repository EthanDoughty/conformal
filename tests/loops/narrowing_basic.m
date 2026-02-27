% Narrowing after widening recovers tighter bounds for loop counters.
% count starts at 0 and increments by 1 each iteration (10 iterations).
% Widening overshoots to [0, 1000]; the narrowing pass should tighten it
% back so that zeros(1, count) can resolve count as a finite symbolic dim.
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: B = matrix[1 x count]
count = 0;
for i = 1:10
    count = count + 1;
end
B = zeros(1, count);
