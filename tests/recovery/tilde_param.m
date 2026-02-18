% Test: ~ as unused parameter in function definition
% EXPECT: warnings = 0
% EXPECT: r1 = matrix[3 x 3]
% EXPECT: r2 = matrix[4 x 1]

function result = tilde_first(~, data)
    result = data;
end

function result = tilde_last(data, ~)
    result = data;
end

r1 = tilde_first(1, zeros(3, 3));
r2 = tilde_last(zeros(4, 1), 99);
