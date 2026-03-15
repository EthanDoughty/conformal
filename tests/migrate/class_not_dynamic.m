function y = check(x)
    t = class(x);
    if class(x) == "double"
        y = ~x;
    else
        y = ~(x > 0);
    end
    s.lambda = 1;
    s.import = 2;
    z = x.(fieldname);
    w = obj.method();
end
