function result = ws_mega_pipeline(A)
    G = ws_gram(A);
    N = ws_normalize(G);
    [T, S] = ws_transform(N);
    r = ws_reduce(T);
    result = ws_scale(r, 2);
end
