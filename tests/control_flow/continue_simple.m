% Test: Continue statement in while loop
% Continue skips to next iteration, doesn't affect shape
% EXPECT: warnings = 0
% EXPECT: B = matrix[5 x 5]
% EXPECT: count = scalar

count = 0;
B = zeros(5, 5);
while count < 20
    count = count + 1;
    if count < 10
        continue;
    end
    B = ones(5, 5);
end
