% Test: If-else where one branch calls unknown function
% Sound behavior: join(matrix[3x3], unknown) = unknown
% This is a precision regression from pre-0.9.3 (was matrix[3x3])
% but is the correct sound choice
% EXPECT: warnings = 1
% EXPECT: A = unknown

if cond
    A = zeros(3, 3);
else
    A = unknown_func();
end
