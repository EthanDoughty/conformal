% Test: nargin interval refinement in if-branch
% Called with 2 args -> nargin interval [2,2]
% if nargin < 2 branch is dead (nargin=2 >= 2)
% EXPECT: warnings = 0
% EXPECT: out = matrix[3 x 1]

function out = myfunc(A, B)
    if nargin < 2
        B = zeros(3, 1);
    end
    out = B;
end

out = myfunc(zeros(3, 1), zeros(3, 1));
