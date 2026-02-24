% Switch with a string case value: no interval refinement, no crash.
% EXPECT: warnings = 0
% EXPECT: x = scalar
mode = 'hello';
switch mode
    case 'hello'
        x = 1;
    otherwise
        x = 2;
end
