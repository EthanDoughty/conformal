% Two-level nesting: outer -> middle -> inner.
% Inner reads middle's local variable via parent scope chain.
% EXPECT: warnings = 0
% EXPECT: result = matrix[3 x 3]

function r = outer()
    A = zeros(3, 3);
    r = middle(A);

    function mr = middle(X)
        B = X;
        mr = inner();

        function ir = inner()
            ir = B;
        end
    end
end

result = outer();
