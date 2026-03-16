function result = feval_regexp(x, pat, rep)
    % feval: dynamic function call
    y = feval('sin', x);
    z = feval(@cos, x, 2);

    % regexp: pattern matching
    m = regexp('hello world', '\w+');
    tokens = regexp('abc123', '\d+', 'match');
    parts = regexp('a,b,c', ',', 'split');

    % regexprep: pattern replacement
    cleaned = regexprep('foo  bar', '\s+', ' ');

    result = y;
end
