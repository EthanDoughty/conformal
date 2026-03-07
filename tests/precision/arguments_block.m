% Test: arguments block parsing (R2019b+)
% The arguments block should be skipped without errors

function r = test()
    arguments
        % empty arguments block
    end
    A = zeros(3, 3);
    r = A;
end
% EXPECT: r: 3x3
