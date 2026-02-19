% Empty matrix [] is identity for concatenation
% EXPECT: warnings = 1

% Horzcat with empty: [[] row] => row
A = [];
B = [A ones(1,3)];
% EXPECT: B = matrix[1 x 3]

% Vertcat with empty: [[] ; col] => col
C = [];
D = [C ; ones(3,1)];
% EXPECT: D = matrix[3 x 1]

% Both sides empty: [[] []] => []
E = [[] []];
% EXPECT: E = matrix[0 x 0]

% Accumulation pattern: start empty, grow via horzcat in loop
% (W_REASSIGN_INCOMPATIBLE fires: acc changes from [0x0] to [2x1])
acc = [];
for i = 1:3
    acc = [acc ones(2,1)];
end
% EXPECT: acc = matrix[2 x 1]
