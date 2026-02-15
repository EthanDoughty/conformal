% Test: String/scalar join across branches
% EXPECT: warnings = 0
% EXPECT: v = unknown

if 1
    v = 'hello';
else
    v = 5;
end
