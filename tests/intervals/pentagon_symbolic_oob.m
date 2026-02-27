% Pentagon symbolic suppression: function param A, n = size(A,1), for i = 1:n.
% i <= n via Pentagon (symbolic bound); A has row dim n via DimEquiv.
% EXPECT: warnings = 0
function test(A)
n = size(A, 1);
for i = 1:n
    x = A(i, 1);
end
end
