% Test: Symbolic colon range dimension extraction.
% EXPECT: warnings = 0
% EXPECT: v = matrix[1 x n]
% EXPECT: w = matrix[1 x 10]
% EXPECT: x = matrix[1 x (n - 2)]
% EXPECT: y = matrix[1 x (n + 1)]
% EXPECT: z = matrix[1 x None]

v = 1:n;            % 1 x n (symbolic endpoint)
w = 1:10;           % 1 x 10 (concrete endpoints)
x = 3:n;            % 1 x (n - 2) = (n - 3) + 1
y = 1:(n+1);        % 1 x (n + 1)
z = 1:2:n;          % stepped range: 1 x None (not handled)
