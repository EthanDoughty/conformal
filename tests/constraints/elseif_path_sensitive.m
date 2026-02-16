% Test: elseif chain — constraint in only one branch should be discarded
% This tests the IfChain handler (not the If handler)
% EXPECT: warnings = 0
A = rand(n, m);
D = rand(p, q);
if n > 5
    E = [A; D];
elseif n > 3
    x = 1;
else
    y = 2;
end
q = 3;
m = 7;
