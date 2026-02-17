function result = ws_with_loop(n)
    result = zeros(1, n);
    for i = 1:n
        result = result + ones(1, n);
    end
end
