% Test end keyword arithmetic in cell indexing
c = {10, 20, 30, 40, 50};

% Simple arithmetic
a = c{end-1};
% EXPECT: a = scalar

b = c{end-2};
% EXPECT: b = scalar

% Nested cells with end arithmetic
outer = {{1, 2, 3}, {4, 5}};
x = outer{1}{end-1};
% EXPECT: x = scalar

% end with symbolic variable (should be unknown)
k = 1;
y = c{end-k};
% EXPECT: y = scalar

% Test with 2D cell array
grid = {1, 2; 3, 4; 5, 6};
z = grid{end-1};
% EXPECT: z = scalar
