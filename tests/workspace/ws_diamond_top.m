function T = ws_diamond_top(A)
    L = ws_diamond_left(A);
    R = ws_diamond_right(A);
    T = ws_add_matrices(L, L);
end
