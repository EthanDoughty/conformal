% Test: cellfun with 'UniformOutput',false returns a cell
% EXPECT: warnings = 0
% EXPECT: result = cell[2 x 3]

C = cell(2, 3);
result = cellfun(@(x) x', C, 'UniformOutput', false);
