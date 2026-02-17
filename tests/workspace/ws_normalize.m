function N = ws_normalize(A)
    s = sum(A);
    N = ws_scale(A, 1);
end
