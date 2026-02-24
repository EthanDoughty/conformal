% Test: end keyword in cell assignment LHS (append pattern)
% EXPECT: warnings = 0

c = {1, 2, 3};
c{end+1} = 4;
c{end} = 99;
