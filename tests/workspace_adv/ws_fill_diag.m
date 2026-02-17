function M = ws_fill_diag(n)
    M = zeros(n, n);
    for i = 1:n
        M(i, i) = 1;
    end
end
