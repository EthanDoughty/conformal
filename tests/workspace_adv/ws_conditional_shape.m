function result = ws_conditional_shape(A)
    if size(A, 1) == size(A, 2)
        result = A * A';
    else
        result = A' * A;
    end
end
