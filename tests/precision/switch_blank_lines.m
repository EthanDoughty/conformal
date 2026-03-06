% Test: switch with blank lines before first case
% Blank lines (comment-only lines stripped by lexer) between switch and case

x = 2;
y = 0;
switch x

    case 1
        y = ones(2, 2);
    case 2
        y = zeros(3, 3);
    otherwise
        y = eye(4);
end
% EXPECT: y: matrix[? x ?]
