% EXPECT: warnings = 0
% EXPECT_FIXPOINT: warnings = 0
% After the while loop, x has been incremented an unknown number of times.
% The interval widening must produce an unbounded upper bound so that a
% subsequent indexing operation does not emit a false W_INDEX_OUT_OF_BOUNDS.
x = 1;
A = zeros(100, 100);
while x < 100
    x = x + 1;
end
% x interval should be widened (upper bound unbounded), no false OOB here
y = A(1, 1);
% EXPECT: y = scalar
