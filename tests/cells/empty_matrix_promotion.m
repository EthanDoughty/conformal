% Test: empty matrix [] promotion to cell array
% Bug B4: x = []; x{1} = 5 triggered W_CELL_ASSIGN_NON_CELL
% because MATLAB's [] is a universal empty initializer

% EXPECT: warnings = 0

x = [];
x{1} = 'hello';
x{2} = 42;
