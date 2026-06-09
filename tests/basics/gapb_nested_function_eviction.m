% Regression: nested functions share parent workspace.
% After calling a nested function that reassigns a parent variable, the parent's
% trusted constant for that variable must be invalidated. No W_ELEMENTWISE_MISMATCH.
% EXPECT: warnings = 0

outer();
function outer()
    L = 0.2;
    inner();
    a = 0:L/400:L/4;
    b = zeros(1, 5);
    c = a .* b;
    function inner()
        L = rand();
    end
end
