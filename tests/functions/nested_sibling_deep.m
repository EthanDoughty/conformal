% Test: Sibling nested functions with their own nested children.
% EXPECT: warnings = 0
% EXPECT: result = matrix[3 x 3]

function result = outer()
    A = zeros(3, 3);
    result = mid1() + mid2();

    function r = mid1()
        r = deep1();
        function dr = deep1()
            dr = A;
        end
    end

    function r = mid2()
        r = deep2();
        function dr = deep2()
            dr = eye(3);
        end
    end
end

result = outer();
