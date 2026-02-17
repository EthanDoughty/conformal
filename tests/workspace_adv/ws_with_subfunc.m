function result = ws_with_subfunc(A)
    result = helper_scale(A, 2);
end

function B = helper_scale(X, k)
    B = X * k;
end
