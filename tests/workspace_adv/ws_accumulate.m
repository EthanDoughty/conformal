function result = ws_accumulate(n)
    result = zeros(0, 3);
    for i = 1:n
        result = [result; ones(1, 3)];
    end
end
