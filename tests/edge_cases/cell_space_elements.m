% Space-separated elements in a cell literal: {1 [] {} 2} is four elements.
% Before the adjacency gate, {} parsed as cell-indexing on [] and vanished,
% leaving a 1x3 cell.

y = {1 [] {} 2};
z = {a{1} 5};

% EXPECT: warnings = 0
% EXPECT: y = cell[1 x 4]
% EXPECT: z = cell[1 x 2]
