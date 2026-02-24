% Two-level nesting: innermost reads variable from outermost (grandparent) scope.
% EXPECT: warnings = 0
% EXPECT: result = matrix[4 x 4]

function r = outer()
    G = zeros(4, 4);
    r = middle();

    function mr = middle()
        mr = inner();

        function ir = inner()
            ir = G;
        end
    end
end

result = outer();
