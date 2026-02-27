% Narrowing tightens interval bounds after conditional accumulation in a loop.
% count starts at 0, increments only when mod condition holds.
% Widening overshoots count to [0, 1000]; narrowing tightens it back so that
% zeros(1, count) resolves count as a finite symbolic dim (not Unknown).
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: B = matrix[1 x count]
count = 0;
for i = 1:20
    if i > 10
        count = count + 1;
    end
end
B = zeros(1, count);
