function r = index_struct_assign(data)
    data(1).name = 'hello';
    data(2).value = 42;
    r = data;
end
