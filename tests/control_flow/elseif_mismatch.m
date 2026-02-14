% Test: Elseif branches with shape mismatch
% Different branch shapes → join produces less precise result
% EXPECT: warnings = 0
% EXPECT: B = matrix[3 x None]

k = 4;
if k == 1
    B = zeros(3, 4);
elseif k == 2
    B = zeros(3, 5);
else
    B = zeros(3, 6);
end
