% Test: stencil access A(i+1) inside for i=1:10 with A having 10 rows should warn.
% The interval of i+1 is [2,11] and A has 10 rows. Pentagon has no bound on i,
% so it cannot suppress. Warning expected.
% EXPECT: warnings = 1
A = zeros(10, 1);
for i = 1:10
    x = A(i+1, 1);
end
