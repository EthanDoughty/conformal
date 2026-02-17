B = mylocal(3);
% EXPECT: B = scalar
% EXPECT: warnings = 0

function result = mylocal(n)
    result = n;
end
