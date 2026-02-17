function D = ws_chain_add(A, B, C)
    temp = ws_add_matrices(A, B);
    D = ws_add_matrices(temp, C);
end
