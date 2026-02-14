% Test: Try/catch with error in try block
% A is 3x4, B is 5x3 → inner dim mismatch (4 != 5), X = unknown in try
% Catch starts from pre-try env, assigns X = matrix[1 x 1]
% Join: unknown (try) + matrix[1 x 1] (catch) = unknown
% EXPECT: warnings = 1
% EXPECT: X = unknown

try
    A = zeros(3, 4);
    B = zeros(5, 3);
    X = A * B;
catch
    X = zeros(1, 1);
end
