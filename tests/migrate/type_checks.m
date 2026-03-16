x = 'hello';
if ischar(x)
    disp('is string');
end
if isnumeric(x)
    disp('is numeric');
end
s = struct('name', 'test', 'value', 42);
disp(s.name);
