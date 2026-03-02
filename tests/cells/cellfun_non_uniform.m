% cellfun where handle returns non-scalar without UniformOutput=false
% MODE: strict
% EXPECT: warnings >= 1
c = {[1 2 3], [4 5 6]};
result = cellfun(@(x) x, c);  % EXPECT_WARNING: W_CELLFUN_NON_UNIFORM
