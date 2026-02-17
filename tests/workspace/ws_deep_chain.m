function D = ws_deep_chain(A)
    D = ws_compose(A, A) + ws_gram(A);
end
