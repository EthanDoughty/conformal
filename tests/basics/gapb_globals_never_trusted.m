% Regression: global-declared variables must never be
% stored as trusted constants, because any function sharing the global can mutate
% the value without the analyzer seeing the write.
%
% EXPECT: warnings = 0

global G
G = 5;
modG();
a = 1:G;
b = [a; ones(1,100)];
% G is kept symbolic (not concrete), so no false mismatch fires.
% The shape of a is symbolic (1 x G) which is also sound.
% EXPECT: b = matrix[2 x None]

function modG()
    global G
    G = 100;
end
