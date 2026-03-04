% Test: 3-level nesting with write-back to outermost scope.
% inner() writes A and B to grandparent scope; outer reads them back.
% EXPECT: warnings = 0
% EXPECT: A = matrix[4 x 4]
% EXPECT: B = matrix[5 x 5]

function [A, B] = outer()
    A = zeros(2, 2);
    B = zeros(3, 3);
    middle();

    function middle()
        inner();

        function inner()
            A = eye(4);
            B = ones(5, 5);
        end
    end
end

[A, B] = outer();
