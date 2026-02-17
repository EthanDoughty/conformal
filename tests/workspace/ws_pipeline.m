function result = ws_pipeline(A)
    [T, S] = ws_transform(A);
    r = ws_reduce(S);
    result = ws_scale(r, 2);
end
