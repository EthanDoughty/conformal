% Two-level nesting: innermost writes to middle scope variable.
% After inner() returns, middle sees the updated value of B.
% EXPECT: warnings = 1
% EXPECT: result = matrix[5 x 5]

function r = outer()
    r = middle();

    function mr = middle()
        B = zeros(2, 2);
        inner();
        mr = B;

        function inner()
            B = zeros(5, 5);
        end
    end
end

result = outer();
