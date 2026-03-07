% Test: arguments block shape extraction (R2019b+)
% Parameter size annotations provide ground-truth shapes for unbound params.
% When process() is called with just X, the optional 'scale' parameter stays
% Bottom, so the arguments-block annotation (1,1) fills it in as Scalar.

A = zeros(3, 4);
r = process(A);

% EXPECT: r = matrix[3 x 4]

function y = process(X, scale)
    arguments
        X (:,:) double
        scale (1,1) double = 1.0
    end
    y = X * scale;
end
