% Test: 4-level nesting with closure read from outermost scope.
% EXPECT: warnings = 0
% EXPECT: result = matrix[5 x 5]

function result = level1()
    A = zeros(5, 5);
    result = level2();

    function r2 = level2()
        r2 = level3();

        function r3 = level3()
            r3 = level4();

            function r4 = level4()
                r4 = A;
            end
        end
    end
end

result = level1();
