% Test: One loop branch calls unknown function, other grows matrix
% Default mode: single iteration through loop, if-join gives unknown, A assigned unknown
% Fixpoint mode: widening sees unknown (top) throughout, A = unknown
% EXPECT: warnings = 2
% EXPECT: A = unknown
% EXPECT_FIXPOINT: warnings = 2
% EXPECT_FIXPOINT: A = unknown

A = zeros(3, 3);
for i = 1:n
    if cond
        A = unknown_func();
    else
        A = [A; zeros(1, 3)];
    end
end
