function net = chain_assign(net, x, l, j)
    net.layers{1}.a{1} = x;
    net.layers{l}.a{j} = x + 1;
    data.results{1}.output(2) = 42;
    data.surface(:) = 0;
end
