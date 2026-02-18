% Cell literal spacing disambiguation
% Same rule as matrix literals: space-before + no-space-after on - means new element

c = {1 -2};
% EXPECT: c = cell[1 x 2]
% EXPECT: warnings = 0
