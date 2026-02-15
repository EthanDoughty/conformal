% Test: Type query functions (iscell, isscalar)
% EXPECT: warnings = 0
% EXPECT: c = cell[2 x 3]
% EXPECT: A = matrix[3 x 3]
% EXPECT: is_cell_c = scalar
% EXPECT: is_cell_A = scalar
% EXPECT: is_scalar_5 = scalar

c = cell(2, 3);
A = zeros(3, 3);
is_cell_c = iscell(c);
is_cell_A = iscell(A);
is_scalar_5 = isscalar(5);
